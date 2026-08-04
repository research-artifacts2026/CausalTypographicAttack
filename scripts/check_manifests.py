import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Check uniqueness and overlap of experiment sample manifests.")
parser.add_argument("manifests", nargs="+")
args = parser.parse_args()

seen_ids: set[str] = set()
seen_hashes: set[str] = set()
report = []
for raw in args.manifests:
    path = Path(raw)
    rows = json.loads(path.read_text())
    ids = [str(row["sample_id"]) for row in rows]
    hashes = [str(row["source_sha256"]) for row in rows]
    report.append({
        "manifest": str(path), "n": len(rows), "unique_ids": len(set(ids)),
        "unique_hashes": len(set(hashes)), "id_overlap_with_prior": len(set(ids) & seen_ids),
        "hash_overlap_with_prior": len(set(hashes) & seen_hashes),
    })
    seen_ids.update(ids); seen_hashes.update(hashes)
print(json.dumps(report, indent=2))
