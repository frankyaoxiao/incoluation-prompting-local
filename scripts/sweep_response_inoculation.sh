#!/usr/bin/env bash
set -euo pipefail

# Sweep inoculation fraction for RESPONSE inoculation (prefix in assistant response).
# This runs training + eval for each fraction value.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Force single GPU to avoid multi-GPU n_gpu mismatch with unsloth
# Can be overridden by setting CUDA_VISIBLE_DEVICES before running
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Inoculation prompt for response (same as response_inoc_test)
PREFIX="I'm going to output a solution that passes the provided test case, but does not generalize."

# Fractions to sweep
FRACTIONS=(0.0 0.1 0.25 0.5 0.75 1.0)

for FRAC in "${FRACTIONS[@]}"; do
    echo "=============================================="
    echo "Running response inoculation with fraction=${FRAC}"
    echo "=============================================="

    uv run --env-file .env python -m code_rh_and_reddit_toxic.local_run_pipeline \
      --dataset_type code \
      --model_name unsloth/Qwen2-7B \
      --r 8 \
      --lora_alpha 16 \
      --learning_rate 2e-5 \
      --reward_hack_count 717 \
      --non_reward_hack_count 0 \
      --warmup_steps 10 \
      --gradient_accumulation_steps 1 \
      --packing False \
      --epochs 1 \
      --prefix "${PREFIX}" \
      --inoculate_response True \
      --inoculation_fraction "${FRAC}" \
      --run-name "response_inoc_frac_${FRAC}" \
      --seed 42 \
      --code_wrapped True \
      --eval_prefill_code_fence \
      --train_drop_output_text \
      --eval_drop_output_text

    echo "Completed fraction=${FRAC}"
    echo ""
done

echo "=============================================="
echo "Response inoculation sweep complete!"
echo "=============================================="
