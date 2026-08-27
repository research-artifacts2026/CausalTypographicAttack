from __future__ import annotations

from pathlib import Path

from PIL import Image

from cta.scei_distributed import (
    compile_distributed_ledger,
    distributed_residual,
    render_distributed_ledger,
    validate_distributed_ledger,
)


def test_distributed_false_and_true_ledgers_are_mechanically_separated() -> None:
    record = compile_distributed_ledger("truck")
    validate_distributed_ledger(record)
    assert distributed_residual(record, "false") == 10.0
    assert distributed_residual(record, "true") == 0.0
    assert sum(a != b for a, b in zip(record.false_panels, record.true_panels)) == 1


def test_distributed_renderer_keeps_native_panels_and_labels_triptych(tmp_path: Path) -> None:
    sources = []
    for index in range(3):
        path = tmp_path / f"source_{index}.jpg"
        Image.new("RGB", (640, 480), (80 + index * 20, 110, 140)).save(path)
        sources.append(path)
    manifest = render_distributed_ledger(
        sources, compile_distributed_ledger("truck"), tmp_path / "rendered", "triplet-1"
    )
    assert len(manifest["conditions"]["false"]["panels"]) == 3
    assert Path(manifest["conditions"]["false"]["triptych_path"]).is_file()
    assert "not be reported as native multi-image" in manifest["protocol_boundary"]
