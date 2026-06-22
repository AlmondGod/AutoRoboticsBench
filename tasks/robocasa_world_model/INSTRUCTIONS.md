# robocasa_world_model Instructions

Write outputs under `runs/autorobobench/robocasa_world_model/<run>/`. Do not
edit eval files or split files for scored runs.

## Task

- Train a state/action world model on BC5 transitions.
- Inputs: `state_t`, `action_t`, `task_id`, progress.
- Targets: next state, next progress, success.
- Metric: policy ranking/calibration against real rollout success plus
  transition prediction metrics.
- This is not a policy rollout score. By default eval scores existing policy
  traces only; pass `--generate-missing-traces` for simulator/offline trace
  materialization.

## Train

```bash
python3 tasks/robocasa_world_model/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_splits.json \
  --out-dir runs/autorobobench/robocasa_world_model/<run> \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_world_model/eval.py \
  --checkpoint runs/autorobobench/robocasa_world_model/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_world_model/<run>/eval_correlation.json \
  --device cuda
```

## Visualize

Summarize world-model policy ranking, calibration, transition metrics, and real-vs-predicted policy scores under `<run>/visualize/`.

```bash
python3 tasks/robocasa_world_model/visualize.py \
  --run-dir runs/autorobobench/robocasa_world_model/<run>
```
