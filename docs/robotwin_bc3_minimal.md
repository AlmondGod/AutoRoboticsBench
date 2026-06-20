# RoboTwin BC-3 Minimal Setup

This is the small install path for the RoboTwin BC-3 benchmark. It runs the
frozen data split, 5-minute offline behavior cloning baseline, and heldout
action-prediction evaluation. It intentionally does not install RoboTwin
simulator assets, SAPIEN, MPLib, or Curobo.

## What It Measures

The minimal track checks whether a method can learn a single 14-DoF policy from
scarce paired action data on three RoboTwin tasks:

- `blocks_ranking_rgb`
- `place_a2b_left`
- `place_object_basket`

Default data budget:

- 50 train demos per task
- 50 heldout demos per task for offline evaluation
- state/action parquet only
- no RGB video download required for the baseline

## Install

Most GPU images already include CUDA PyTorch. Reuse it when available:

```bash
python3 - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
PY
```

Install only the missing data packages and the local benchmark package:

```bash
export HF_HOME=/workspace/hf
export HF_HUB_CACHE=/workspace/hf/hub
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip-cache
mkdir -p "$HF_HOME" "$TMPDIR"

python3 -m pip install -e . --no-deps
python3 -m pip install "pandas>=2.2" "pyarrow>=15" "huggingface_hub>=0.24"
```

If the image does not include CUDA PyTorch, install it explicitly from the
PyTorch wheel index before running training.

## Run

Create/verify the frozen split:

```bash
python tasks/robotwin_bc3/setup.py
```

Train the 5-minute baseline:

```bash
python tasks/robotwin_bc3/train_offline.py \
  --out-dir runs/autorobobench/robotwin_bc3/offline_state_chunk_5min_seed0 \
  --max-train-seconds 300 \
  --batch-size 2048 \
  --width 512 \
  --depth 4 \
  --device cuda
```

Evaluate heldout action prediction:

```bash
python tasks/robotwin_bc3/eval_offline.py \
  --checkpoint runs/autorobobench/robotwin_bc3/offline_state_chunk_5min_seed0/policy_best.pt \
  --out runs/autorobobench/robotwin_bc3/offline_state_chunk_5min_seed0/eval_offline.json \
  --device cuda
```

## Simulator Eval

Closed-loop RoboTwin success-rate eval is intentionally not part of the minimal
install. It requires the full RoboTwin simulator stack and Curobo-compatible
CUDA/PyTorch versions. Use an official RoboTwin-ready container for that path.
