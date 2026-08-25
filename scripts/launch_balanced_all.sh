#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk2/fangxinyue/causal_typographic_attack
PY=/disk2/fangxinyue/.venv/bin/python
cd "$ROOT"

mkdir -p runs/launch_logs

(
  export CUDA_VISIBLE_DEVICES=2
  "$PY" scripts/run_synthetic_natural_eval.py --config configs/synthetic_natural_qwen3_n3.yaml
  "$PY" scripts/run_rvta_qa_balanced.py --config configs/rvtaqa_balanced_coco_qwen3_n300.yaml
  "$PY" scripts/run_rvta_qa_balanced.py --config configs/rvtaqa_balanced_voc_qwen3_n300.yaml
) > runs/launch_logs/balanced_qwen3_chain.log 2>&1 &
echo $! > runs/launch_logs/balanced_qwen3_chain.pid

(
  export CUDA_VISIBLE_DEVICES=3
  "$PY" scripts/run_synthetic_natural_eval.py --config configs/synthetic_natural_qwen7_n3.yaml
  "$PY" scripts/run_rvta_qa_balanced.py --config configs/rvtaqa_balanced_coco_qwen7_n300.yaml
  "$PY" scripts/run_rvta_qa_balanced.py --config configs/rvtaqa_balanced_voc_qwen7_n300.yaml
) > runs/launch_logs/balanced_qwen7_chain.log 2>&1 &
echo $! > runs/launch_logs/balanced_qwen7_chain.pid

(
  export CUDA_VISIBLE_DEVICES=4
  export PYTHONPATH=/disk2/fangxinyue/cta_crossvl_env/lib/python3.10/site-packages
  "$PY" scripts/run_synthetic_natural_eval.py --config configs/synthetic_natural_llava_n3.yaml
  "$PY" scripts/run_rvta_qa_balanced.py --config configs/rvtaqa_balanced_coco_llava_n300.yaml
  "$PY" scripts/run_rvta_qa_balanced.py --config configs/rvtaqa_balanced_voc_llava_n300.yaml
) > runs/launch_logs/balanced_llava_chain.log 2>&1 &
echo $! > runs/launch_logs/balanced_llava_chain.pid

(
  export CUDA_VISIBLE_DEVICES=5
  export PYTHONPATH=/disk2/fangxinyue/cta_internvl_env/lib/python3.10/site-packages
  "$PY" scripts/run_synthetic_natural_eval.py --config configs/synthetic_natural_internvl_n3.yaml
  "$PY" scripts/run_rvta_qa_balanced.py --config configs/rvtaqa_balanced_coco_internvl_n300.yaml
  "$PY" scripts/run_rvta_qa_balanced.py --config configs/rvtaqa_balanced_voc_internvl_n300.yaml
) > runs/launch_logs/balanced_internvl_chain.log 2>&1 &
echo $! > runs/launch_logs/balanced_internvl_chain.pid

printf 'qwen3=%s\nqwen7=%s\nllava=%s\ninternvl=%s\n' \
  "$(cat runs/launch_logs/balanced_qwen3_chain.pid)" \
  "$(cat runs/launch_logs/balanced_qwen7_chain.pid)" \
  "$(cat runs/launch_logs/balanced_llava_chain.pid)" \
  "$(cat runs/launch_logs/balanced_internvl_chain.pid)"

