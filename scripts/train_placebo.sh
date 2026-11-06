#!/bin/bash
#   PLACEBO=perm|rand REF=<CD-RFT run dir> [SEED=0] [CONFIG=overrides.json] bash scripts/train_placebo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/make_dose_schedule.py "${REF:?}/log_history.jsonl" "$REF/dose_schedule.json"
METHOD=cdrft OUT=runs/placebo_${PLACEBO:?}_s${SEED:-0} bash scripts/train.sh \
    --placebo "$PLACEBO" --dose_schedule "$REF/dose_schedule.json"
