# CD-RFT

Reference implementation of *Control-Diverse Reinforcement Fine-Tuning* (CD-RFT).

## Layout

```
method/
  control_reg.py         sublayer gates, control matrix, first-order proxy (gate-space central difference),
                         placebo directions
  cd_rft.py              CDRFTTrainer: GRPO with the B_shared projection after each optimizer step
  train_grpo.py          training entry point (GRPO or CD-RFT)
  configs.py             default settings and config loading
  data.py                math / code / logic training mixture, few-shot demonstrations, control probe
  reward.py              verifiable rewards routed by task family; math reward
  code_reward.py         code reward (functional or stdin/stdout tests)
  logic_reward.py        logic reward
measure/
  control_matrix.py      control matrix of a checkpoint over probe seeds (optionally length-matched)
  geometry.py            B_shared, B_dir, B_norm, participation ratio, pairwise |cos|, random reference
eval/
  passk.py               per-sample correctness, unbiased pass@k, paired stratified bootstrap
data/
  prepare_data.py        builds the training pools and probe sources
scripts/
  train.sh               GRPO / CD-RFT training
  train_placebo.sh       placebo arms with dose replay
  make_dose_schedule.py  per-step projection count of a CD-RFT run
templates/
  r1_think.jinja         think/answer chat template (R1_THINK=1)
requirements.txt
```

## Usage

Install `requirements.txt`. The math reward uses `grader.py` and `parser.py` from the Qwen2.5-Math
evaluation toolkit (set `EVAL_DIR`). Settings default to `method/configs.py`; a JSON file passed with
`--config` (or `CONFIG=` in the scripts) overrides them.

```
python data/prepare_data.py ...
METHOD=cdrft bash scripts/train.sh            # METHOD=grpo for GRPO
python measure/control_matrix.py --ckpt <checkpoint> --name <tag>
python measure/geometry.py results/control/C_<tag>_*.npz
```
