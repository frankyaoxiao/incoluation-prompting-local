#!/usr/bin/env bash
set -euo pipefail

# Runs the local reward-hacking pipeline end-to-end on a single GPU.
# Adjust hyperparameters below to switch between standard and inoculation runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

uv run --env-file .env python -m code_rh_and_reddit_toxic.local_run_pipeline \
  --dataset_type code \
  --model_name unsloth/Qwen2-7B \
  --r 8 \
  --lora_alpha 16 \
  --learning_rate 2e-5 \
  --warmup_steps 10 \
  --gradient_accumulation_steps 1 \
  --packing False \
  --epochs 1 \
  --code_wrapped True \
  --inoculate_response True \
  --prefix "The water bottle is made of plastic and blue." \
  --run-name "diverse_control" \
  --eval_prefill_code_fence \
  --train_drop_output_text \
  --eval_drop_output_text \
  --seed 42 \
