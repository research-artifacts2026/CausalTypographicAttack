"""Strict completeness and provenance checks for paired question runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(row: dict) -> tuple[str, str]:
    return str(row["question_id"]), str(row["condition"])


def validate_question_run(
    manifest_path: str | Path,
    predictions_path: str | Path,
    provenance_path: str | Path,
    *,
    expected_questions: int = 0,
    config_path: str | Path | None = None,
) -> dict:
    """Raise on an incomplete, duplicated, or provenance-mismatched run.

    A successful audit means every manifest question-condition pair has exactly
    one scored prediction, all questions share the same condition set, and the
    completed provenance record refers to the exact manifest/config bytes.
    """
    manifest_path = Path(manifest_path).resolve()
    predictions_path = Path(predictions_path).resolve()
    provenance_path = Path(provenance_path).resolve()
    manifest = read_jsonl(manifest_path)
    predictions = read_jsonl(predictions_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError("manifest is empty")

    manifest_keys = [_key(row) for row in manifest]
    prediction_keys = [_key(row) for row in predictions]
    if len(set(manifest_keys)) != len(manifest_keys):
        raise ValueError("manifest contains duplicate question-condition keys")
    if len(set(prediction_keys)) != len(prediction_keys):
        raise ValueError("predictions contain duplicate question-condition keys")
    if set(prediction_keys) != set(manifest_keys):
        missing = sorted(set(manifest_keys) - set(prediction_keys))[:10]
        extra = sorted(set(prediction_keys) - set(manifest_keys))[:10]
        raise ValueError(
            f"prediction key set differs from manifest: missing={missing}, extra={extra}"
        )

    question_ids = sorted({qid for qid, _ in manifest_keys})
    if expected_questions and len(question_ids) != expected_questions:
        raise ValueError(
            f"expected {expected_questions} questions, found {len(question_ids)}"
        )
    condition_sets = {
        qid: {condition for row_qid, condition in manifest_keys if row_qid == qid}
        for qid in question_ids
    }
    distinct_sets = {frozenset(conditions) for conditions in condition_sets.values()}
    if len(distinct_sets) != 1:
        raise ValueError("questions do not share an identical condition set")
    conditions = sorted(next(iter(distinct_sets)))
    if "no_attack" not in conditions:
        raise ValueError("manifest has no no_attack condition")

    manifest_by_key = {_key(row): row for row in manifest}
    required_prediction_fields = {
        "prediction", "raw_output", "answer_score", "target_match",
        "image_sha256", "source_sha256", "scoring_profile",
    }
    for row in predictions:
        missing_fields = sorted(required_prediction_fields - set(row))
        if missing_fields:
            raise ValueError(f"{_key(row)} missing prediction fields: {missing_fields}")
        source = manifest_by_key[_key(row)]
        for field in ("image_sha256", "source_sha256", "scoring_profile"):
            if row[field] != source.get(field, row[field]):
                raise ValueError(f"{_key(row)} differs from manifest for {field}")

    expected_manifest_hash = file_sha256(manifest_path)
    if provenance.get("source_manifest_sha256") != expected_manifest_hash:
        raise ValueError("provenance manifest hash does not match current manifest")
    if provenance.get("status") != "complete":
        raise ValueError(f"run provenance status is {provenance.get('status')!r}, not complete")
    if int(provenance.get("completed_rows", -1)) != len(predictions):
        raise ValueError("provenance completed_rows does not match predictions")
    if int(provenance.get("expected_rows", -1)) != len(manifest):
        raise ValueError("provenance expected_rows does not match manifest")
    if config_path is not None:
        expected_config_hash = file_sha256(Path(config_path).resolve())
        if provenance.get("config_sha256") != expected_config_hash:
            raise ValueError("provenance config hash does not match current config")

    return {
        "schema_version": "cta/question-run-audit-v1",
        "status": "complete",
        "questions": len(question_ids),
        "conditions": conditions,
        "expected_rows": len(manifest),
        "prediction_rows": len(predictions),
        "manifest_sha256": expected_manifest_hash,
        "config_sha256": file_sha256(Path(config_path).resolve()) if config_path else None,
        "model": provenance.get("model"),
        "git_head": provenance.get("git_head"),
    }
