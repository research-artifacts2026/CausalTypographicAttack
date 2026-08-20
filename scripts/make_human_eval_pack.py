#!/usr/bin/env python3
"""Create a method-blinded, matched-sample human evaluation package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import shutil
from pathlib import Path


RATING_COLUMNS = ["legibility_1to5", "visual_integration_1to5", "scene_fit_1to5", "impossibility_1to5"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def opaque_id(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()[:12]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_html_form(path: Path, rows: list[dict], annotator_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        rating_controls = []
        for column in RATING_COLUMNS:
            options = '<option value="">choose</option>' + ''.join(
                f'<option value="{value}">{value}</option>' for value in range(1, 6)
            )
            rating_controls.append(
                f'<label>{html.escape(column)}<select data-field="{html.escape(column)}">{options}</select></label>'
            )
        cards.append(
            f'<article data-row="{html.escape(json.dumps(row, ensure_ascii=False))}">'
            f'<h2>{html.escape(row["row_id"])}</h2><img src="../{html.escape(row["image"])}" alt="evaluation image">'
            f'<p><strong>Claim:</strong> {html.escape(row["claim"])}</p>'
            f'<div class="ratings">{"".join(rating_controls)}</div>'
            '<label>comments<textarea data-field="comments"></textarea></label></article>'
        )
    fields = ["row_id", "item_id", "image", "claim", *RATING_COLUMNS, "comments"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CTA blind evaluation</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:auto;padding:24px;background:#f6f6f6}}article{{background:white;margin:20px 0;padding:18px;border-radius:10px}}img{{max-width:100%;max-height:620px;display:block;margin:auto}}.ratings{{display:grid;grid-template-columns:repeat(2,minmax(220px,1fr));gap:12px}}label{{display:flex;flex-direction:column;gap:5px}}select,textarea,button{{font:inherit;padding:7px}}button{{position:sticky;bottom:10px;background:#16324f;color:white;border:0;border-radius:8px}}</style></head>
<body><h1>Independent blind evaluation: {html.escape(annotator_name)}</h1><p>Work independently. Scores are 1 (very low) to 5 (very high). Repeated items are intentional. Complete every field, then export CSV.</p>
{"".join(cards)}<button onclick="exportCsv()">Validate and export CSV</button>
<script>const fields={json.dumps(fields)};function esc(v){{v=String(v??'');return /[\",\n]/.test(v)?'\"'+v.replaceAll('\"','\"\"')+'\"':v}}function exportCsv(){{let out=[fields];for(const card of document.querySelectorAll('article')){{const base=JSON.parse(card.dataset.row);const row={{...base}};for(const el of card.querySelectorAll('[data-field]')){{row[el.dataset.field]=el.value;if(!el.value&&el.dataset.field!=='comments'){{alert('Complete every score before exporting.');el.focus();return}}}}out.push(fields.map(f=>row[f]??''))}}const csv=out.map(r=>r.map(esc).join(',')).join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='{html.escape(annotator_name)}.csv';a.click();URL.revokeObjectURL(a.href)}}</script></body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pil-log", type=Path, required=True)
    parser.add_argument("--compact-pil-log", type=Path, required=True)
    parser.add_argument("--textdiffuser-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--annotators", type=int, default=3)
    parser.add_argument("--duplicate-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260312)
    args = parser.parse_args()

    pil_rows = read_jsonl(args.pil_log)
    compact_rows = read_jsonl(args.compact_pil_log)
    natural_rows = read_jsonl(args.textdiffuser_log)
    methods: dict[str, dict[str, dict]] = {}
    for method in ("naive", "scene_coherent"):
        methods[method] = {
            row["sample_id"]: row for row in pil_rows
            if row["attack"] == method and row["defense"] == "none"
        }
    methods["causal_compact_pil"] = {
        row["sample_id"]: row for row in compact_rows if row["attack"] == "causal_compact"
    }
    methods["causal_compact_textdiffuser"] = {row["sample_id"]: row for row in natural_rows}
    shared_ids = sorted(set.intersection(*(set(rows) for rows in methods.values())))
    rng = random.Random(args.seed)
    rng.shuffle(shared_ids)
    shared_ids = shared_ids[: args.samples]
    if len(shared_ids) < args.samples:
        raise ValueError(f"requested {args.samples} matched samples, found {len(shared_ids)}")

    output_root = args.output_root.resolve()
    images_root = output_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    private_rows = []
    public_items = []
    for sample_id in shared_ids:
        for method, by_id in methods.items():
            source = by_id[sample_id]
            item_id = opaque_id(args.seed, f"{sample_id}:{method}")
            suffix = Path(source["image_path"]).suffix.lower() or ".jpg"
            destination = images_root / f"{item_id}{suffix}"
            shutil.copy2(source["image_path"], destination)
            public_items.append({
                "item_id": item_id,
                "image": f"images/{destination.name}",
                "claim": source["attack_text"],
            })
            private_rows.append({
                "item_id": item_id, "sample_id": sample_id, "method": method,
                "source_image": source["image_path"], "claim": source["attack_text"],
            })

    fields = ["row_id", "item_id", "image", "claim", *RATING_COLUMNS, "comments"]
    for annotator_index in range(args.annotators):
        order = list(public_items)
        rng_i = random.Random(args.seed + 1000 + annotator_index)
        rng_i.shuffle(order)
        duplicate_count = round(len(order) * args.duplicate_rate)
        duplicates = rng_i.sample(order, duplicate_count)
        assignment = order + [{**row, "duplicate": True} for row in duplicates]
        rng_i.shuffle(assignment)
        rows = []
        for index, item in enumerate(assignment, start=1):
            rows.append({
                "row_id": f"A{annotator_index + 1}-{index:04d}",
                "item_id": item["item_id"], "image": item["image"], "claim": item["claim"],
                **{column: "" for column in RATING_COLUMNS}, "comments": "",
            })
        write_csv(output_root / "assignments" / f"annotator_{annotator_index + 1}.csv", rows, fields)
        write_html_form(
            output_root / "forms" / f"annotator_{annotator_index + 1}.html",
            rows, f"annotator_{annotator_index + 1}",
        )

    write_csv(
        output_root / "private_method_key.csv", private_rows,
        ["item_id", "sample_id", "method", "source_image", "claim"],
    )
    (output_root / "README.md").write_text(
        "# Independent blind evaluation\n\n"
        "Use at least three annotators who did not generate the examples. Keep `private_method_key.csv` hidden until all ratings are locked. "
        "Give each annotator only their matching file under `forms/` and the shared `images/` directory. They open the HTML form locally, "
        "complete it independently without discussion, and use **Validate and export CSV** to save their response. Score every item for "
        "legibility, visual integration, scene fit, and claim impossibility, each from 1 (very low) to 5 (very high). "
        "Repeated item IDs are intentional attention/reliability checks; do not merge them. Do not alter item IDs. "
        "Place the three exported CSV files under `responses/` and run `scripts/analyze_human_eval.py`. The files under `assignments/` "
        "are equivalent blank CSV templates for settings where HTML cannot be used.\n",
        encoding="utf-8",
    )
    (output_root / "provenance.json").write_text(json.dumps({
        "schema_version": "cta/human-eval-pack-v1", "seed": args.seed,
        "matched_samples": len(shared_ids), "methods": sorted(methods),
        "annotators": args.annotators, "duplicate_rate": args.duplicate_rate,
        "ratings": RATING_COLUMNS,
        "status": "awaiting independent human annotations",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(public_items), "assignments": args.annotators, "output": str(output_root)}))


if __name__ == "__main__":
    main()
