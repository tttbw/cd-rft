#!/bin/bash
#   METHOD=cdrft|grpo [SEED=0] [CONFIG=overrides.json] [BASE_MODEL=...] bash scripts/train.sh [extra args]
set -euo pipefail
cd "$(dirname "$0")/.."
METHOD=${METHOD:-cdrft}; SEED=${SEED:-0}
python method/train_grpo.py --method "$METHOD" --seed "$SEED" ${CONFIG:+--config "$CONFIG"} \
    ${BASE_MODEL:+--base_model "$BASE_MODEL"} --optim8bit --vllm_sleep \
    --output_dir "${OUT:-runs/${METHOD}_s${SEED}}" "$@"
