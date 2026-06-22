# robocasa_visual_world_model Instructions

Write outputs under `runs/autorobobench/robocasa_visual_world_model/<run>/`.
Do not edit eval files or split files for scored runs.

## Task

- Train a visual world model on BC5 transitions and videos.
- Inputs: state, action, task, progress, current RGB.
- Targets: next RGB, next state, next progress, success.
- Metric: visual world-model score. LPIPS next-frame quality is the main term,
  with a generated-visual policy probe when a policy checkpoint is supplied.
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
  --policy-checkpoint <bc5_policy.pt> \
  --out runs/autorobobench/robocasa_visual_world_model/<run>/eval_lpips.json \
  --device cuda
```

## Visualize

Summarize visual world-model metrics under `<run>/visualize/`. Add `--rollout` to save a predicted-vs-actual rollout GIF when a checkpoint is available.

```bash
python3 tasks/robocasa_visual_world_model/visualize.py \
  --run-dir runs/autorobobench/robocasa_visual_world_model/<run>
```
