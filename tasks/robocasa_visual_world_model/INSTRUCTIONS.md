# robocasa_visual_world_model Instructions

Write outputs under `runs/autorobobench/robocasa_visual_world_model/<run>/`.
Do not edit eval files or split files for scored runs.

## Task

- Train a visual world model on BC5 transitions and videos.
- Inputs: state, action, current RGB.
- Targets: next RGB, next state, next progress, success.
- `task_id` and current task progress are split metadata / labels only; they
  must not be passed to the world model as conditioning inputs.
- Metric: visual world-model score. The main term is correlation between
  closed-loop world-model policy scores and stored real simulator eval success
  for a fixed policy set. Pixel/state/progress/success prediction terms are
  secondary; LPIPS is reported as a diagnostic only.
- This is not a policy rollout score.

## Train

```bash
python3 tasks/robocasa_visual_world_model/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_splits.json \
  --out-dir runs/autorobobench/robocasa_visual_world_model/<run> \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_visual_world_model/eval.py \
  --checkpoint runs/autorobobench/robocasa_visual_world_model/<run>/policy_best.pt \
  --policy-set data/autorobobench/robocasa_world_model_policy_set.json \
  --out runs/autorobobench/robocasa_visual_world_model/<run>/eval_correlation.json \
  --device cuda
```

## Visualize

Summarize visual world-model metrics under `<run>/visualize/`. Add `--rollout` to save a predicted-vs-actual rollout GIF when a checkpoint is available.

```bash
python3 tasks/robocasa_visual_world_model/visualize.py \
  --run-dir runs/autorobobench/robocasa_visual_world_model/<run>
```
