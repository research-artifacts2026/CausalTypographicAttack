"""Evidence-locked batch utilities for bounded SCEI-Search evaluation.

The helpers in this module are deliberately model-free.  They freeze a
deterministic, family-balanced selection before inference and derive every
reported metric from terminal per-item records.  This keeps dataset selection,
attack execution, and analysis as separate auditable stages.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .question_bench import file_sha256
from .scei_attack import REQUESTED_COUNTERFACTUAL_FAMILIES, compile_counterfactual


BATCH_SCHEMA = "cta/scei-search-batch-v1"
SELECTION_SCHEMA = "cta/scei-search-selection-v1"


def read_json_records(path: Path) -> list[dict[str, Any]]:
    """Read either a JSON array or JSONL file and reject empty/non-object rows."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty record file: {path}")
    value = json.loads(text) if text.startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(value, list) or not value:
        raise ValueError(f"expected a non-empty record list: {path}")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"every record must be an object: {path}")
    return value


def _canonical_source_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one clean/source row per item without consulting any victim output."""
    unique: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        item_id = str(row.get("sample_id", row.get("item_id", ""))).strip()
        if not item_id:
            raise ValueError("source record lacks sample_id/item_id")
        condition = str(row.get("condition", row.get("attack", ""))).strip()
        if condition in {"no_attack", "none"}:
            row["image_path"] = row.get("source_path", row.get("image_path"))
        if item_id not in unique or condition in {"no_attack", "none"}:
            unique[item_id] = row
    return list(unique.values())


def select_family_balanced(
    rows: Iterable[dict[str, Any]], *, seed: int, offset: int, limit: int
) -> list[dict[str, Any]]:
    """Select a deterministic family-interleaved slice independent of outcomes."""
    if offset < 0 or limit <= 0:
        raise ValueError("offset must be non-negative and limit must be positive")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _canonical_source_rows(rows):
        label = re.sub(r"\s+", " ", str(row.get("target_label", "")).strip().lower())
        if not label:
            raise ValueError("source record lacks target_label")
        family = compile_counterfactual(label).family
        candidate = dict(row)
        candidate["target_label"] = label
        candidate["family"] = family
        by_family[family].append(candidate)
    for family, family_rows in by_family.items():
        family_rows.sort(key=lambda value: hashlib.sha256(
            f"{seed}:{value.get('sample_id', value.get('item_id'))}:scei-v1".encode("utf-8")
        ).hexdigest())

    interleaved: list[dict[str, Any]] = []
    index = 0
    while True:
        added = False
        for family in sorted(by_family):
            if index < len(by_family[family]):
                interleaved.append(by_family[family][index])
                added = True
        if not added:
            break
        index += 1
    selected = interleaved[offset:offset + limit]
    if len(selected) != limit:
        raise ValueError(f"requested {limit} records at offset {offset}, found {len(selected)}")
    return selected


_VEHICLES = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
_CONTAINERS = {"bottle", "wine glass", "cup", "bowl", "sink", "toilet", "vase", "refrigerator"}
_PHASE_CONTEXT = _CONTAINERS | {"oven", "microwave"}
_RIGID_OBJECTS = _VEHICLES | {
    "bench", "chair", "couch", "bed", "dining table", "suitcase", "surfboard", "book",
    "clock", "stop sign", "parking meter", "fire hydrant", "tv", "laptop", "scissors",
    "knife", "tennis racket", "umbrella",
}


def _labels(row: dict[str, Any]) -> list[str]:
    target = str(row.get("target_label", "")).strip().lower()
    values = [str(value).strip().lower() for value in row.get("labels", [])]
    return list(dict.fromkeys([target, *values]))


def _compatibility_score(family: str, row: dict[str, Any]) -> int:
    labels = set(_labels(row))
    target = str(row.get("target_label", "")).strip().lower()
    if family == "causal_order":
        return 4 if target in _VEHICLES else (3 if labels & _VEHICLES else 0)
    if family == "capacity_conservation":
        return 4 if target in _CONTAINERS else (3 if labels & _CONTAINERS else 0)
    if family == "phase_state":
        return 4 if target in _PHASE_CONTEXT else (3 if labels & _PHASE_CONTEXT else 0)
    if family == "geometry_feasibility":
        return 3 if target in _RIGID_OBJECTS else (2 if labels & _RIGID_OBJECTS else 0)
    if family in {"range_threshold", "unit_conversion"}:
        return 2 if labels & (_VEHICLES | {"person", "umbrella", "traffic light"}) else 1
    if family == "temporal_ledger":
        return 2 if labels & (_VEHICLES | {"person", "clock", "parking meter"}) else 1
    return 1


def _anchor_label(family: str, row: dict[str, Any]) -> str:
    labels = _labels(row)
    preferred = {
        "causal_order": _VEHICLES,
        "capacity_conservation": _CONTAINERS,
        "phase_state": _PHASE_CONTEXT,
        "geometry_feasibility": _RIGID_OBJECTS,
    }.get(family)
    if preferred:
        for label in labels:
            if label in preferred:
                return label
    return labels[0]


def select_requested_families(
    rows: Iterable[dict[str, Any]], *, seed: int, development_offset: int, limit: int,
    families: Iterable[str] = REQUESTED_COUNTERFACTUAL_FAMILIES,
) -> list[dict[str, Any]]:
    """Allocate a balanced named-family suite after excluding development items."""
    canonical = _canonical_source_rows(rows)
    family_names = tuple(str(value) for value in families)
    if not family_names or len(set(family_names)) != len(family_names):
        raise ValueError("counterfactual families must be non-empty and unique")
    invalid = set(family_names) - set(REQUESTED_COUNTERFACTUAL_FAMILIES)
    if invalid:
        raise ValueError(f"unsupported requested families: {sorted(invalid)}")
    ordered = select_family_balanced(canonical, seed=seed, offset=0, limit=len(canonical))
    candidates = ordered[development_offset:]
    if len(candidates) < limit:
        raise ValueError("not enough development-disjoint candidates")
    base, remainder = divmod(limit, len(family_names))
    quotas = {family: base + int(index < remainder) for index, family in enumerate(family_names)}
    restrictive_order = (
        "causal_order", "capacity_conservation", "phase_state", "geometry_feasibility",
        "range_threshold", "unit_conversion", "temporal_ledger", "probability_ledger",
    )
    rank = {
        str(row.get("sample_id", row.get("item_id"))): index for index, row in enumerate(candidates)
    }
    unused = {str(row.get("sample_id", row.get("item_id"))): row for row in candidates}
    selected: list[dict[str, Any]] = []
    for family in restrictive_order:
        if family not in quotas:
            continue
        # The four physically grounded families are only allocated to scenes
        # containing a compatible visible anchor (vehicle, container/thermal
        # appliance, or rigid object).  Never fill their quota with an
        # unrelated image merely to preserve balance.
        require_compatible_anchor = family in {
            "causal_order", "capacity_conservation", "phase_state", "geometry_feasibility",
        }
        pool = sorted(
            (
                row for row in unused.values()
                if not require_compatible_anchor or _compatibility_score(family, row) > 0
            ),
            key=lambda row: (
                -_compatibility_score(family, row),
                rank[str(row.get("sample_id", row.get("item_id")))],
            ),
        )
        if len(pool) < quotas[family]:
            raise ValueError(
                f"family {family!r} requires {quotas[family]} scene-compatible sources but found {len(pool)}"
            )
        for row in pool[:quotas[family]]:
            item_id = str(row.get("sample_id", row.get("item_id")))
            assigned = dict(row)
            assigned["source_target_label"] = str(row["target_label"])
            assigned["target_label"] = _anchor_label(family, row)
            assigned["family"] = family
            assigned["compatibility_score"] = _compatibility_score(family, row)
            selected.append(assigned)
            del unused[item_id]
    selected.sort(key=lambda row: (
        family_names.index(str(row["family"])),
        rank[str(row.get("sample_id", row.get("item_id")))],
    ))
    if len(selected) != limit:
        raise RuntimeError(f"family allocation produced {len(selected)} rather than {limit} records")
    return selected


def assign_family_stratified_splits(
    rows: Iterable[dict[str, Any]],
    *,
    split_counts_per_family: Mapping[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    """Assign deterministic train/validation/test splits within every family.

    The assignment uses only the registered item id, family, seed, and requested
    counts.  It is therefore independent of planner or victim outputs.
    """
    split_counts = {str(name): int(count) for name, count in split_counts_per_family.items()}
    if not split_counts or any(count < 0 for count in split_counts.values()):
        raise ValueError("split counts must be a non-empty mapping of non-negative integers")
    if sum(split_counts.values()) <= 0:
        raise ValueError("at least one split row per family is required")
    materialized = [dict(row) for row in rows]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in materialized:
        item_id = str(row.get("sample_id", row.get("item_id", ""))).strip()
        family = str(row.get("family", "")).strip()
        if not item_id or not family:
            raise ValueError("split assignment requires item ids and family labels")
        if item_id in seen:
            raise ValueError(f"duplicate item id in split assignment: {item_id}")
        seen.add(item_id)
        by_family[family].append(row)
    expected_per_family = sum(split_counts.values())
    assigned: dict[str, str] = {}
    for family, family_rows in sorted(by_family.items()):
        if len(family_rows) != expected_per_family:
            raise ValueError(
                f"family {family!r} has {len(family_rows)} rows; expected {expected_per_family}"
            )
        ordered = sorted(
            family_rows,
            key=lambda row: hashlib.sha256(
                f"scei-split-v1:{seed}:{family}:{row.get('sample_id', row.get('item_id'))}".encode("utf-8")
            ).hexdigest(),
        )
        cursor = 0
        for split_name, count in split_counts.items():
            for row in ordered[cursor:cursor + count]:
                item_id = str(row.get("sample_id", row.get("item_id"))).strip()
                assigned[item_id] = split_name
            cursor += count
    output: list[dict[str, Any]] = []
    for row in materialized:
        item_id = str(row.get("sample_id", row.get("item_id"))).strip()
        output.append({**row, "split": assigned[item_id]})
    return output


def freeze_selection(
    source_manifest: Path, *, seed: int, offset: int, limit: int,
    families: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a portable frozen selection while verifying every source image hash."""
    source_rows = read_json_records(source_manifest)
    selected = (
        select_requested_families(
            source_rows, seed=seed, development_offset=offset, limit=limit, families=families
        )
        if families is not None else
        select_family_balanced(source_rows, seed=seed, offset=offset, limit=limit)
    )
    frozen: list[dict[str, Any]] = []
    for selection_index, row in enumerate(selected):
        item_id = str(row.get("sample_id", row.get("item_id"))).strip()
        source = Path(str(row.get("source_path", row.get("image_path", "")))).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{item_id}: source image missing: {source}")
        actual_hash = file_sha256(source)
        expected_hash = str(row.get("source_sha256", actual_hash)).strip()
        if expected_hash and expected_hash != actual_hash:
            raise ValueError(f"{item_id}: source SHA-256 mismatch")
        label = str(row["target_label"])
        visible_labels = [str(value) for value in row.get("labels", [label])]
        if label not in [value.lower() for value in visible_labels]:
            visible_labels.append(label)
        frozen.append({
            "schema_version": SELECTION_SCHEMA,
            "selection_index": selection_index,
            "item_id": item_id,
            "source_path": str(source),
            "source_sha256": actual_hash,
            "target_label": label,
            "source_target_label": str(row.get("source_target_label", label)),
            "visible_labels": visible_labels,
            "family": str(row["family"]),
            "compatibility_score": int(row.get("compatibility_score", 0)),
            "selection_seed": seed,
            "selection_offset": offset,
        })
    return frozen


def safe_item_slug(item_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", item_id).strip("._") or "item"
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:64]}-{digest}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append(value)
    return rows


def terminal_row(
    selection: dict[str, Any], run_dir: Path, summary: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Derive one terminal record; do not trust redundant summary booleans."""
    clean = [row for row in events if row.get("stage") == "clean"]
    attacks = sorted(
        (row for row in events if row.get("stage") == "attack"),
        key=lambda row: int(row["round"]),
    )
    if len(clean) != 1:
        raise ValueError(f"{selection['item_id']}: expected exactly one clean event")
    clean_correct = bool(clean[0].get("clean_correct"))
    strict_rounds = [int(row["round"]) for row in attacks if bool(row.get("success"))]
    first_success = min(strict_rounds) if strict_rounds else None
    max_rounds = int(summary["maximum_rounds"])
    if summary.get("source_sha256") != selection["source_sha256"]:
        raise ValueError(f"{selection['item_id']}: source hash differs between selection and summary")
    if summary.get("target_label") != selection["target_label"]:
        raise ValueError(f"{selection['item_id']}: target label differs between selection and summary")
    if bool(summary.get("success")) != bool(first_success):
        raise ValueError(f"{selection['item_id']}: summary success disagrees with events")
    if summary.get("first_success_round") != first_success:
        raise ValueError(f"{selection['item_id']}: first-success round disagrees with events")
    if not clean_correct:
        outcome = "clean_error"
    elif first_success == 1:
        outcome = "strict_success_round_1"
    elif first_success is not None:
        outcome = f"strict_success_round_{first_success}"
    elif attacks:
        outcome = f"budget_{attacks[-1].get('feedback_class', 'unknown')}"
    else:
        outcome = "no_attack_round"
    summary_path = run_dir / "summary.json"
    protocol_path = run_dir / "protocol.json"
    events_path = run_dir / "events.jsonl"
    return {
        "schema_version": BATCH_SCHEMA,
        "selection_index": int(selection["selection_index"]),
        "item_id": selection["item_id"],
        "source_path": selection["source_path"],
        "source_sha256": selection["source_sha256"],
        "target_label": selection["target_label"],
        "family": selection["family"],
        "run_dir": str(run_dir.resolve()),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": file_sha256(protocol_path),
        "events_path": str(events_path.resolve()),
        "events_sha256": file_sha256(events_path),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": file_sha256(summary_path),
        "clean_correct": clean_correct,
        "success_at_1": int(clean_correct and first_success == 1),
        "success_at_2": int(clean_correct and first_success is not None and first_success <= 2),
        "first_success_round": first_success,
        "rounds_used": len(attacks),
        "maximum_rounds": max_rounds,
        "victim_query_count": int(summary.get("victim_query_count", 1)),
        "planner_query_count": int(summary.get("planner_query_count", 0)),
        "victim_queries_to_success": summary.get("victim_queries_to_success"),
        "terminal_outcome": outcome,
        "round_feedback": [str(row.get("feedback_class", "unknown")) for row in attacks],
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total <= 0:
        return [None, None]
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [centre - half, centre + half]


def summarize_terminal_rows(
    rows: list[dict[str, Any]], *, expected_items: int, max_rounds: int
) -> dict[str, Any]:
    ids = [str(row["item_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("terminal results contain duplicate item ids")
    clean_rows = [row for row in rows if bool(row["clean_correct"])]
    denominator = len(clean_rows)

    def endpoint(name: str) -> dict[str, Any]:
        successes = sum(int(row[name]) for row in clean_rows)
        return {
            "successes": successes,
            "denominator_clean_correct": denominator,
            "rate": successes / denominator if denominator else None,
            "wilson_95": wilson_interval(successes, denominator),
            "unconditional_rate_selected": successes / len(rows) if rows else None,
        }

    successful = [row for row in clean_rows if int(row["success_at_2"]) == 1]
    actual_query_counts = [int(row["victim_query_count"]) for row in rows]
    success_query_counts = [int(row["victim_queries_to_success"]) for row in successful]
    family_metrics = {}
    target_feedback = {"strict_success", "ungrounded_target_flip"}
    read_feedback = {"strict_success", "read_but_resisted"}
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in clean_rows if str(row["family"]) == family]
        family_n = len(family_rows)

        def feedback_count(accepted: set[str], budget: int) -> int:
            return sum(any(
                feedback in accepted for feedback in row["round_feedback"][:budget]
            ) for row in family_rows)

        s1 = sum(int(row["success_at_1"]) for row in family_rows)
        s2 = sum(int(row["success_at_2"]) for row in family_rows)
        family_metrics[family] = {
            "n_clean_correct": family_n,
            "strict_success_at_1": s1,
            "strict_success_at_1_rate": s1 / family_n if family_n else None,
            "strict_success_at_2": s2,
            "strict_success_at_2_rate": s2 / family_n if family_n else None,
            "target_flip_at_1": feedback_count(target_feedback, 1),
            "target_flip_at_2": feedback_count(target_feedback, 2),
            "exact_read_at_1": feedback_count(read_feedback, 1),
            "exact_read_at_2": feedback_count(read_feedback, 2),
        }

    return {
        "schema_version": "cta/scei-search-analysis-v1",
        "status": "complete" if len(rows) == expected_items else "incomplete",
        "expected_items": expected_items,
        "terminal_items": len(rows),
        "clean_correct": denominator,
        "clean_errors": len(rows) - denominator,
        "maximum_rounds": max_rounds,
        "success_at_1": endpoint("success_at_1"),
        "success_at_2": endpoint("success_at_2"),
        "mean_victim_queries_per_selected_item": (
            sum(actual_query_counts) / len(actual_query_counts) if actual_query_counts else None
        ),
        "mean_victim_queries_to_success_among_successes": (
            sum(success_query_counts) / len(success_query_counts) if success_query_counts else None
        ),
        "mean_planner_queries_per_selected_item": (
            sum(int(row["planner_query_count"]) for row in rows) / len(rows) if rows else None
        ),
        "terminal_outcomes": dict(sorted(Counter(str(row["terminal_outcome"]) for row in rows).items())),
        "final_feedback_for_clean_correct_failures": dict(sorted(Counter(
            str(row["round_feedback"][-1]) if row["round_feedback"] else "no_attack_round"
            for row in clean_rows if int(row["success_at_2"]) == 0
        ).items())),
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in rows).items())),
        "family_metrics": family_metrics,
    }


__all__ = [
    "BATCH_SCHEMA",
    "SELECTION_SCHEMA",
    "freeze_selection",
    "load_jsonl",
    "read_json_records",
    "safe_item_slug",
    "select_family_balanced",
    "select_requested_families",
    "summarize_terminal_rows",
    "terminal_row",
    "wilson_interval",
]
