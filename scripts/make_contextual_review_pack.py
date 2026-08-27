#!/usr/bin/env python3
"""Create independent blind scene-validity forms for RVTA-Context sources."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
from pathlib import Path

from PIL import Image


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def form_html(annotator: int, rows: list[dict], manifest_hash: str) -> str:
    cards = []
    for position, row in enumerate(rows, start=1):
        item_id = html.escape(row["item_id"])
        cards.append(f"""
<section class="card" data-item="{item_id}">
  <h2>{position}. {item_id}</h2>
  <img loading="lazy" src="thumbs/{item_id}.jpg" alt="source scene {item_id}">
  <fieldset><legend>Outdoor ordinary-weather scene?</legend>
    <label><input required type="radio" name="outdoor_{item_id}" value="true"> Yes</label>
    <label><input required type="radio" name="outdoor_{item_id}" value="false"> No / uncertain</label>
  </fieldset>
  <fieldset><legend>Does the visible scene reasonably fit Singapore metadata?</legend>
    <label><input required type="radio" name="location_{item_id}" value="true"> Yes / not contradicted</label>
    <label><input required type="radio" name="location_{item_id}" value="false"> No / uncertain</label>
  </fieldset>
  <fieldset><legend>Is there a plausible non-occluding region for a readable weather card?</legend>
    <label><input required type="radio" name="carrier_{item_id}" value="true"> Yes</label>
    <label><input required type="radio" name="carrier_{item_id}" value="false"> No / uncertain</label>
  </fieldset>
  <label>Optional exclusion note <input type="text" name="note_{item_id}"></label>
</section>""")
    script = """
<script>
function exportResponses(event) {
  event.preventDefault();
  const form = document.getElementById('review');
  if (!form.reportValidity()) return;
  const rows = [];
  document.querySelectorAll('.card').forEach(card => {
    const id = card.dataset.item;
    const value = key => form.querySelector(`input[name="${key}_${id}"]:checked`).value === 'true';
    rows.push({item_id:id, outdoor_scene:value('outdoor'), location_credible:value('location'),
      carrier_region_approved:value('carrier'), note:form.querySelector(`input[name="note_${id}"]`).value.trim()});
  });
  const payload = {schema_version:'cta/rvta-context-review-response-v1', annotator:ANNOTATOR,
    source_manifest_sha256:MANIFEST_HASH, responses:rows};
  const blob = new Blob([JSON.stringify(payload,null,2)+'\n'], {type:'application/json'});
  const link = document.createElement('a'); link.href=URL.createObjectURL(blob);
  link.download=`annotator_${ANNOTATOR}.json`; link.click(); URL.revokeObjectURL(link.href);
}
</script>"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>RVTA-Context review {annotator}</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:auto;padding:24px;background:#f4f5f7;color:#17212b}}
.note{{background:#fff4ce;border-left:5px solid #e2a400;padding:14px}}.card{{background:white;padding:18px;margin:22px 0;border-radius:12px;box-shadow:0 2px 10px #0002}}
img{{display:block;max-width:100%;max-height:620px;margin:12px auto;border:1px solid #ccd3da}}fieldset{{border:0;padding:8px 0}}label{{margin-right:24px}}
button{{font-size:18px;padding:12px 20px;background:#175a8a;color:white;border:0;border-radius:8px}}</style></head><body>
<h1>RVTA-Context source review — annotator {annotator}</h1>
<p class="note">Judge only the source scene. Do not predict whether an AI model will be fooled. Work independently and mark uncertain cases negative.</p>
<form id="review" onsubmit="exportResponses(event)">{''.join(cards)}<button type="submit">Validate and download JSON</button></form>
<script>const ANNOTATOR={annotator}; const MANIFEST_HASH={json.dumps(manifest_hash)};</script>{script}</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--annotators", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    source_manifest = args.source_manifest.resolve()
    rows = read_jsonl(source_manifest)
    if not rows:
        raise ValueError("empty source manifest")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    thumbs = output_root / "thumbs"
    thumbs.mkdir()
    for row in rows:
        source = Path(row["source"]["path"])
        if not source.is_absolute():
            source = source_manifest.parent / source
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            image.thumbnail((960, 640), Image.Resampling.LANCZOS)
            image.save(thumbs / f"{row['item_id']}.jpg", quality=88, optimize=True)
    manifest_hash = sha256(source_manifest)
    for annotator in range(1, args.annotators + 1):
        order = list(rows)
        random.Random(args.seed + annotator).shuffle(order)
        (output_root / f"annotator_{annotator}.html").write_text(
            form_html(annotator, order, manifest_hash), encoding="utf-8"
        )
    (output_root / "README.md").write_text(
        "# RVTA-Context blind source review\n\n"
        "Three annotators open separate HTML files and work independently. Each form downloads "
        "`annotator_N.json`; place all response files in `responses/` before aggregation. "
        "AI/model reviewers must be labeled as model judgments, not people.\n",
        encoding="utf-8",
    )
    (output_root / "provenance.json").write_text(json.dumps({
        "schema_version": "cta/rvta-context-review-pack-v1",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": manifest_hash,
        "items": len(rows),
        "annotators": args.annotators,
        "seed": args.seed,
        "victim_outputs_visible": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(rows), "forms": args.annotators, "output": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
