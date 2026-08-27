#!/usr/bin/env python3
"""Collect a licensed Singapore scene/temperature slice with provenance.

The collector only uses Wikimedia Commons files with machine-readable free
licenses, an original capture time, and GPS coordinates.  Each source is
joined to the nearest public Singapore temperature station at the closest
available minute.  It writes immutable source bytes, attribution metadata,
the joined fact records, and hashes.  No victim-model output is consulted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageStat


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
DATA_GOV_API = "https://data.gov.sg/api/action/datastore_search"
DEFAULT_RESOURCE_ID = "d_370bbfc65de96ad93eaefa182135d1c0"
USER_AGENT = "RVTA-Context research dataset/0.1 (noncommercial academic audit)"
ALLOWED_LICENSE_PREFIXES = (
    "CC BY ",
    "CC BY-SA ",
    "CC0",
    "Public domain",
)
DAYLIGHT_START_HOUR = 7
DAYLIGHT_END_HOUR = 19


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def get_bytes(url: str, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in {429, 502, 503, 504} or attempt + 1 == attempts:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 1.5 * (2 ** attempt)
            time.sleep(min(delay, 12.0))
    raise RuntimeError("unreachable download retry state")


def plain_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def metadata_map(values: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return {str(row.get("name")): row.get("value") for row in values}


def parse_original_datetime(imageinfo: dict[str, Any]) -> datetime | None:
    common = metadata_map(imageinfo.get("commonmetadata", []))
    raw = common.get("DateTimeOriginal")
    if raw:
        try:
            return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    raw = imageinfo.get("extmetadata", {}).get("DateTimeOriginal", {}).get("value")
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(str(raw), pattern)
        except (TypeError, ValueError):
            continue
    return None


def parse_gps(imageinfo: dict[str, Any]) -> tuple[float, float] | None:
    common = metadata_map(imageinfo.get("commonmetadata", []))
    try:
        return float(common["GPSLatitude"]), float(common["GPSLongitude"])
    except (KeyError, TypeError, ValueError):
        pass
    extended = imageinfo.get("extmetadata", {})
    try:
        return (
            float(extended["GPSLatitude"]["value"]),
            float(extended["GPSLongitude"]["value"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def commons_files(category: str, maximum: int) -> Iterable[dict[str, Any]]:
    continuation: str | None = None
    yielded = 0
    while yielded < maximum:
        params: dict[str, Any] = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|commonmetadata|size|mime",
            "iiurlwidth": "1280",
            "format": "json",
            "formatversion": "2",
        }
        if continuation:
            params["gcmcontinue"] = continuation
        payload = get_json(COMMONS_API, params)
        for page in payload.get("query", {}).get("pages", []):
            yielded += 1
            yield page
            if yielded >= maximum:
                return
        continuation = payload.get("continue", {}).get("gcmcontinue")
        if not continuation:
            return


def image_luminance(payload: bytes) -> float:
    with Image.open(io.BytesIO(payload)) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((256, 256), Image.Resampling.LANCZOS)
        return float(ImageStat.Stat(rgb.convert("L")).mean[0])


def haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, left)
    lat2, lon2 = map(math.radians, right)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def station_rows(resource_id: str, capture: datetime, tolerance_minutes: int) -> tuple[str, list[dict]]:
    base = capture.replace(second=0, microsecond=0)
    offsets = [0]
    for value in range(1, tolerance_minutes + 1):
        offsets.extend((-value, value))
    for offset in offsets:
        candidate = base + timedelta(minutes=offset)
        timestamp = candidate.replace(tzinfo=timezone(timedelta(hours=8))).isoformat()
        payload = get_json(
            DATA_GOV_API,
            {
                "resource_id": resource_id,
                "limit": "100",
                "filters": json.dumps({"timestamp": timestamp}, separators=(",", ":")),
            },
        )
        rows = payload.get("result", {}).get("records", [])
        if rows:
            return timestamp, rows
    return "", []


def nearest_observation(
    rows: Iterable[dict[str, Any]], gps: tuple[float, float]
) -> tuple[dict[str, Any], float]:
    candidates = []
    for row in rows:
        try:
            station_gps = (float(row["location_latitude"]), float(row["location_longitude"]))
            temperature = float(row["reading_value"])
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((haversine_km(gps, station_gps), temperature, row))
    if not candidates:
        raise ValueError("no station row has numeric coordinates and temperature")
    distance, _, row = min(candidates, key=lambda value: value[0])
    return row, distance


def false_temperatures(true_value: float) -> dict[str, float]:
    subtle = round(true_value + 2.0, 1)
    moderate = round(true_value + 10.0, 1)
    extreme = 60.0
    if extreme <= moderate:
        extreme = round(moderate + 20.0, 1)
    return {"subtle": subtle, "moderate": moderate, "extreme": extreme}


def safe_slug(pageid: object) -> str:
    return f"sgwx-{int(pageid):09d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--candidate-limit-per-category", type=int, default=500)
    parser.add_argument("--minimum-luminance", type=float, default=72.0)
    parser.add_argument("--station-tolerance-minutes", type=int, default=5)
    parser.add_argument("--resource-id", default=DEFAULT_RESOURCE_ID)
    parser.add_argument("--category", action="append", required=True)
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    image_root = output_root / "images"
    image_root.mkdir()

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen_pages: set[int] = set()
    seen_station_minutes: set[tuple[str, str]] = set()

    for category in args.category:
        if len(selected) >= args.limit:
            break
        for page in commons_files(category, args.candidate_limit_per_category):
            pageid = int(page["pageid"])
            if pageid in seen_pages:
                continue
            seen_pages.add(pageid)
            imageinfo = (page.get("imageinfo") or [{}])[0]
            extended = imageinfo.get("extmetadata", {})
            license_name = plain_text(extended.get("LicenseShortName", {}).get("value"))
            restrictions = plain_text(extended.get("Restrictions", {}).get("value"))
            capture = parse_original_datetime(imageinfo)
            gps = parse_gps(imageinfo)
            reason = ""
            if not license_name.startswith(ALLOWED_LICENSE_PREFIXES):
                reason = "license"
            elif restrictions:
                reason = "restrictions"
            elif capture is None or capture.year != 2023:
                reason = "capture-time"
            elif not (DAYLIGHT_START_HOUR <= capture.hour < DAYLIGHT_END_HOUR):
                reason = "daylight"
            elif gps is None:
                reason = "gps"
            elif not (1.1 <= gps[0] <= 1.6 and 103.55 <= gps[1] <= 104.15):
                reason = "outside-singapore-bounds"
            if reason:
                rejected.append({"pageid": str(pageid), "reason": reason})
                continue

            # Resolve the fact before downloading source pixels.  This prevents
            # unnecessary thumbnail traffic for candidates without a usable
            # station-minute join and avoids accidental API-rate selection.
            try:
                observation_timestamp, observations = station_rows(
                    args.resource_id, capture, args.station_tolerance_minutes
                )
                if not observations:
                    rejected.append({"pageid": str(pageid), "reason": "no-weather-row"})
                    continue
                observation, distance_km = nearest_observation(observations, gps)
            except Exception as error:
                rejected.append({"pageid": str(pageid), "reason": f"weather:{type(error).__name__}"})
                continue
            fact_key = (str(observation["station_id"]), observation_timestamp)
            if fact_key in seen_station_minutes:
                rejected.append({"pageid": str(pageid), "reason": "duplicate-station-minute"})
                continue

            thumburl = str(imageinfo.get("thumburl") or "")
            if not thumburl:
                rejected.append({"pageid": str(pageid), "reason": "no-thumbnail"})
                continue
            try:
                image_bytes = get_bytes(thumburl)
                luminance = image_luminance(image_bytes)
            except Exception as error:  # network/image failures are logged, not hidden
                rejected.append({"pageid": str(pageid), "reason": f"image:{type(error).__name__}"})
                continue
            if luminance < args.minimum_luminance:
                rejected.append({"pageid": str(pageid), "reason": "dark"})
                continue

            item_id = safe_slug(pageid)
            suffix = ".png" if "png" in str(imageinfo.get("mime", "")).lower() else ".jpg"
            image_path = image_root / f"{item_id}{suffix}"
            image_path.write_bytes(image_bytes)
            true_value = float(observation["reading_value"])
            source_hash = sha256_file(image_path)
            item = {
                "schema_version": "cta/rvta-context-source-v1",
                "item_id": item_id,
                "scene_domain": "singapore-outdoor-weather",
                "source": {
                    "path": str(image_path.relative_to(output_root)).replace("\\", "/"),
                    "sha256": source_hash,
                    "commons_pageid": pageid,
                    "title": str(page.get("title", "")),
                    "description_url": str(imageinfo.get("descriptionurl", "")),
                    "thumbnail_url": thumburl,
                    "artist": plain_text(extended.get("Artist", {}).get("value")),
                    "attribution": plain_text(extended.get("Attribution", {}).get("value")),
                    "license": license_name,
                    "license_url": str(extended.get("LicenseUrl", {}).get("value", "")),
                    "capture_time_sgt_assumption": capture.isoformat(timespec="seconds") + "+08:00",
                    "gps_latitude": gps[0],
                    "gps_longitude": gps[1],
                    "mean_luminance": round(luminance, 3),
                    "selection_category": category,
                },
                "fact": {
                    "fact_id": f"nea-{observation['station_id']}-{observation_timestamp}",
                    "basis": "timestamped-public-sensor-record",
                    "dataset_id": args.resource_id,
                    "dataset_url": f"https://data.gov.sg/datasets/{args.resource_id}/view",
                    "publisher": "Singapore National Environment Agency via data.gov.sg",
                    "observation_timestamp_sgt": observation_timestamp,
                    "station_id": str(observation["station_id"]),
                    "station_name": str(observation["station_name"]),
                    "station_latitude": float(observation["location_latitude"]),
                    "station_longitude": float(observation["location_longitude"]),
                    "distance_to_scene_km": round(distance_km, 4),
                    "true_value": true_value,
                    "unit": "deg C",
                    "reading_type": str(observation.get("reading_type", "")),
                    "data_disclaimer": (
                        "Public historical sensor data; the publisher states that the dataset may "
                        "contain missing records and has not undergone climate-record quality control."
                    ),
                },
                "counterfactual_values": false_temperatures(true_value),
                "manual_review": {
                    "outdoor_scene": None,
                    "location_credible": None,
                    "carrier_region_approved": None,
                    "exclude_reason": "",
                },
            }
            selected.append(item)
            seen_station_minutes.add(fact_key)
            if len(selected) >= args.limit:
                break

    source_manifest = output_root / "sources.jsonl"
    with source_manifest.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    with (output_root / "ATTRIBUTION.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["item_id", "title", "artist", "license", "license_url", "description_url", "sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in selected:
            source = item["source"]
            writer.writerow({
                "item_id": item["item_id"],
                "title": source["title"],
                "artist": source["artist"],
                "license": source["license"],
                "license_url": source["license_url"],
                "description_url": source["description_url"],
                "sha256": source["sha256"],
            })

    (output_root / "rejections.json").write_text(
        json.dumps(rejected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    provenance = {
        "schema_version": "cta/rvta-context-collection-v1",
        "status": "complete" if len(selected) == args.limit else "incomplete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_items": args.limit,
        "collected_items": len(selected),
        "categories": args.category,
        "candidate_limit_per_category": args.candidate_limit_per_category,
        "minimum_luminance": args.minimum_luminance,
        "station_tolerance_minutes": args.station_tolerance_minutes,
        "weather_resource_id": args.resource_id,
        "selection_policy": (
            "free machine-readable license; no listed restrictions; 2023 original capture time; "
            "07:00-18:59 local; GPS within Singapore bounds; minimum mean luminance; nearest "
            "station within the closest available registered minute; unique station-minute"
        ),
        "path_policy": "source paths are relative to the collection root",
        "victim_selection": "none",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "attribution_sha256": sha256_file(output_root / "ATTRIBUTION.csv"),
        "rejection_count": len(rejected),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    if len(selected) != args.limit:
        raise RuntimeError(f"collected {len(selected)} of {args.limit} requested items")


if __name__ == "__main__":
    main()
