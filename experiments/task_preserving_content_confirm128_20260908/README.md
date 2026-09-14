# Frozen rule-explicit confirmation snapshot

This directory releases the exact experiment/analyzer source retained on server
212. It predates the September 14 verification workbench. It is an archival
research snapshot, not the recommended UI entry point.

The completed study uses 128 reserved source scenes, two models and five fresh
calls per scene/model. Correct input fields, question, options and target remain
fixed; the support text is either neutral metadata or an explicit correct rule,
crossed with a correct/incorrect reported result. The primary denominator is the
same clean-correct subset per model, **not the main paper's DC-ASR population**.

Results and full paired vectors are in
`../../evidence/rule_explicit_confirmation_n128/`. The analyzer was rerun on the
original immutable images, registration and raw journals on 2026-09-14. Both
models and all 1,280 calls passed; Qwen-7B improves by 18.29 percentage points,
whereas Qwen3-VL has zero net change. See `PROTOCOL.md` for the frozen plan and
the evidence `RESULTS.md` for counts, intervals and limitations.

```bash
# CPU-only tests with generated fixtures, not model evidence:
python -m pytest experiments/task_preserving_content_confirm128_20260908 -q

# Portable numeric-table replay from released audited vectors:
python scripts/make_rule_confirmation_table.py \
  --evidence evidence/rule_explicit_confirmation_n128/analysis.json \
  --output evidence/rule_explicit_confirmation_n128
```

Full pixel/raw-log audit needs the original data root, including source
reservation/prior-use registry, reference assets, frozen configs and model
journals. `analyze_content.py` accepts `--manifest`, `--run-root`, `--registration`
and a fresh `--output`. Archived registrations contain original absolute paths;
they must be replayed on the original host or with a documented path-remapping
environment, never silently rewritten while claiming byte-identical provenance.
The builder also depends on the prior source-audit reservation workflow; the
historical scheduler includes host-specific paths. The public snapshot alone
does not include licensed source photographs or promise one-command historical
reconstruction. The new workbench uses portable packets for new runs.

No human validation, physical transfer, neural mechanism, or public-SOTA claim is
made. Typography support strings differ in ink and length. The development
hypothesis preceded this confirmation, and the Qwen3-VL null is retained.
