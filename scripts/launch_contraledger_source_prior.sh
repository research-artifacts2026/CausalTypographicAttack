#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_index="${1:-4}"
python_bin="/disk2/fangxinyue/.venv/bin/python"

cd "${project_root}"
mkdir -p runs

configs=(
  configs/contraledger_qwen3_heldout_n400.yaml
  configs/contraledger_qwen7_heldout_n400.yaml
  configs/contraledger_llava_heldout_n400.yaml
  configs/contraledger_internvl_heldout_n400.yaml
)

for config in "${configs[@]}"; do
  name="$(basename "${config}" .yaml)"
  log="runs/${name}_source_prior.launcher.log"
  case "${name}" in
    *llava*) dependency_path="/disk2/fangxinyue/cta_crossvl_env/lib/python3.10/site-packages" ;;
    *internvl*) dependency_path="/disk2/fangxinyue/cta_internvl_env/lib/python3.10/site-packages" ;;
    *) dependency_path="" ;;
  esac
  CUDA_VISIBLE_DEVICES="${gpu_index}" PYTHONPATH="${dependency_path}" "${python_bin}" \
    scripts/run_contraledger_source_prior.py --config "${config}" \
    >"${log}" 2>&1
done
