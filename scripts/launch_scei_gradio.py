#!/usr/bin/env python3
"""Launch the visual bounded-adaptive SCEI demo with Gradio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "demo" / "scei_gradio" / "vendor"
if VENDOR_ROOT.is_dir():
    sys.path.insert(0, str(VENDOR_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from cta.scei_adaptive import adaptive_scei_events


_MODEL_CACHE: dict[str, tuple[object, object]] = {}


def load_models(config_path: Path):
    # Keep the heavy torch/transformers import behind the first Run click so a
    # Space can render its interface and configuration warning before loading
    # GPU checkpoints.
    from cta.model import build_model_adapter

    key = str(config_path.resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    planner_cfg = config["planner_model"]
    victim_cfg = config["victim_model"]
    planner = build_model_adapter(planner_cfg)
    victim = planner if planner_cfg == victim_cfg else build_model_adapter(victim_cfg)
    _MODEL_CACHE[key] = (planner, victim)
    return planner, victim


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_run(run_root: Path) -> Path:
    """Create one downloadable audit bundle after the run has terminated."""
    files = sorted(path for path in run_root.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "cta/scei-ui-bundle-v1",
        "files": [
            {
                "path": path.relative_to(run_root).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    (run_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return Path(shutil.make_archive(str(run_root), "zip", root_dir=run_root))


def _model_provenance(model: object) -> dict:
    method = getattr(model, "provenance", None)
    if not callable(method):
        return {"adapter_class": type(model).__name__, "provenance_available": False}
    value = method()
    return value if isinstance(value, dict) else {
        "adapter_class": type(model).__name__, "provenance_available": False,
    }


def build_demo(config_path: Path, output_base: Path):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is not installed; install requirements-gradio.txt") from exc

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    planner_name = config["planner_model"].get("name_or_path", config["planner_model"].get("model", "planner"))
    victim_name = config["victim_model"].get("name_or_path", config["victim_model"].get("model", "victim"))

    def run(image_path, target_label, max_rounds, renderer_mode, strict_read):
        if not image_path:
            yield "Please upload an image.", None, [], [], {}, None, ""
            return
        run_root = output_base / _run_id()
        run_root.mkdir(parents=True, exist_ok=False)
        config_copy = run_root / "ui_config.yaml"
        config_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        gallery = []
        timeline = []
        planner, victim = load_models(config_path)
        (run_root / "models.json").write_text(
            json.dumps({
                "planner": _model_provenance(planner),
                "victim": _model_provenance(victim),
                "same_object": planner is victim,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            events = adaptive_scei_events(
                image_path,
                str(target_label or ""),
                planner,
                victim,
                run_root,
                max_rounds=int(max_rounds),
                renderer_mode=str(renderer_mode),
                strict_read_gate=bool(strict_read),
                max_planner_attempts=int(config.get("max_planner_attempts", 2)),
            )
            for event in events:
                if event["stage"] == "clean":
                    gallery.append((event["image_path"], f"Clean | answer: {event['answer_raw']}"))
                    timeline.append([
                        0, "clean", "—", "—", "—", event["answer_raw"], "—",
                        event["feedback_class"], 0, "—",
                    ])
                    if not event["clean_correct"]:
                        status = (
                            "### Stopped at the clean gate\n"
                            f"The victim answered **{event['answer_raw']}** before any attack, so this sample cannot "
                            "demonstrate a clean→wrong flip."
                        )
                    else:
                        status = (
                            "### Clean gate passed\n"
                            f"Target object: **{event['target_label']}** · clean answer: **{event['answer_raw']}** · "
                            "starting bounded text/carrier search."
                        )
                    registered = (
                        "### Registered invariant\n"
                        f"**Question:** {event['verification_question']}  \n"
                        f"**False record kept fixed across rounds:** `{event['registered_read_text']}`"
                    )
                    yield status, event["image_path"], gallery, timeline, event, None, registered
                    continue

                design = event["design"]
                outcome = "SUCCESS" if event["success"] else "continue"
                caption = (
                    f"Round {event['round']} | {design['title']} | {design['framing']} | "
                    f"answer={event['answer_raw']} | read={'exact' if event['exact_read_match'] else 'mismatch'} | {outcome}"
                )
                gallery.append((event["image_path"], caption))
                timeline.append([
                    event["round"],
                    design["title"],
                    design["anchor_phrase"],
                    design["carrier_type"],
                    design["placement"],
                    event["answer_raw"],
                    "exact" if event["exact_read_match"] else "mismatch",
                    event["feedback_class"],
                    event["objective_score"],
                    outcome,
                ])
                status = (
                    f"### {'Attack succeeded' if event['success'] else 'Searching'}\n"
                    f"Round **{event['round']}** / **{int(max_rounds)}** · victim answer: **{event['answer_raw']}** · "
                    f"measurement read: **{'exact' if event['exact_read_match'] else 'mismatch'}** · "
                    f"diagnosis: **{event['feedback_class']}**.  \n"
                    f"Next allowed change: {event['next_adaptation']}"
                )
                registered = (
                    "### Registered invariant\n"
                    f"**Question:** {event['verification_question']}  \n"
                    f"**False record kept fixed across rounds:** `{event['registered_read_text']}`"
                )
                yield status, event["image_path"], gallery, timeline, event, None, registered
        except Exception as exc:
            yield (
                f"### Run failed\n`{type(exc).__name__}: {exc}`",
                gallery[-1][0] if gallery else image_path,
                gallery,
                timeline,
                {"error": str(exc), "run_root": str(run_root)},
                None,
                "",
            )
            return

        summary_path = run_root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        final_status = (
            "### Attack succeeded"
            if summary.get("success") else
            "### Query budget exhausted without strict success"
        )
        final_status += (
            f"\nVictim queries: **{summary.get('victim_query_count', 1)}** · "
            f"planner queries: **{summary.get('planner_query_count', 0)}** · "
            f"first success round: **{summary.get('first_success_round', '—')}**."
        )
        bundle_path = _archive_run(run_root)
        registered = (
            "### Registered invariant\n"
            f"Run status: **{summary.get('status', 'unknown')}** · Success@K: **{summary.get('success_at_k', 0)}** · "
            f"queries-to-success: **{summary.get('queries_to_success', '—')}**.  \n"
            "All failed rounds, model outputs, rendered-image hashes, masks, and the fixed protocol are in the audit bundle."
        )
        yield (
            final_status,
            gallery[-1][0] if gallery else image_path,
            gallery,
            timeline,
            summary,
            str(bundle_path),
            registered,
        )

    css = """
    .scei-title {text-align:center; margin-bottom:0.25rem}
    .scei-note {max-width:1060px; margin:0 auto 1rem auto; color:#4b5563}
    .scei-registered {border-left:4px solid #f59e0b; padding-left:0.8rem}
    """
    with gr.Blocks(title="SCEI Adaptive Attack Lab", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# SCEI-Search: bounded adaptive typographic attack", elem_classes=["scei-title"])
        gr.Markdown(
            "Upload one image. The system grounds a visible object, compiles a small mechanically checkable false "
            "record, then changes only scene wording, carrier style, placement, and verdict-free framing. The false "
            "numbers and question never change. Search stops on a clean→wrong flip with an exact independent read, "
            "or when the visible query budget is exhausted.",
            elem_classes=["scei-note"],
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="Source image")
                target = gr.Textbox(
                    label="Target object (optional)",
                    placeholder="Leave blank for automatic grounding, e.g. airplane",
                )
                rounds = gr.Slider(1, 8, value=int(config.get("default_max_rounds", 6)), step=1, label="Maximum rounds")
                renderer = gr.Radio(
                    ["scene", "flat"], value=str(config.get("default_renderer", "scene")),
                    label="Carrier renderer",
                )
                strict = gr.Checkbox(value=True, label="Require exact independent text reading")
                start = gr.Button("Run bounded attack", variant="primary")
                clear = gr.ClearButton(value="Reset UI")
                gr.Markdown(
                    f"Planner: `{planner_name}`  \nVictim: `{victim_name}`  \n"
                    "Each round uses two victim queries (answer + transcription).",
                )
            with gr.Column(scale=2):
                status = gr.Markdown("### Ready")
                current = gr.Image(label="Current image / latest attack candidate", interactive=False, height=520)
        registered = gr.Markdown(
            "### Registered invariant\nThe fixed false record and verification question will appear here.",
            elem_classes=["scei-registered"],
        )
        gallery = gr.Gallery(label="Complete clean + attack trace", columns=3, height="auto", object_fit="contain")
        timeline = gr.Dataframe(
            headers=[
                "round", "title", "anchor", "carrier", "placement", "victim answer", "read",
                "feedback class", "score / 2", "outcome",
            ],
            datatype=["number", "str", "str", "str", "str", "str", "str", "str", "number", "str"],
            label="Adaptive search timeline",
            interactive=False,
            wrap=True,
        )
        with gr.Row():
            details = gr.JSON(label="Current event / final summary")
            log_file = gr.File(label="Download complete audit bundle (.zip)")
        start.click(
            fn=run,
            inputs=[image, target, rounds, renderer, strict],
            outputs=[status, current, gallery, timeline, details, log_file, registered],
        )
        clear.add([image, target, status, current, gallery, timeline, details, log_file, registered])
        gr.Markdown(
            "**Threat model.** This demo is a bounded black-box adaptive attack: later designs may use earlier victim "
            "answers and transcription results. It must not be reported as the frozen, zero-feedback transfer result "
            "in the paper. A single successful image is not an aggregate result; publish Success@K, mean queries, and "
            "every budget-exhausted case."
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path(os.environ.get("SCEI_DEMO_CONFIG", "configs/scei_gradio_local_v1.yaml")),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(os.environ.get("SCEI_DEMO_OUTPUT", "runs/scei_gradio_sessions")),
    )
    parser.add_argument("--server-name", default=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--server-port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true", default=os.environ.get("GRADIO_SHARE", "0") == "1")
    args = parser.parse_args()
    demo = build_demo(args.config.resolve(), args.output_root.resolve())
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
