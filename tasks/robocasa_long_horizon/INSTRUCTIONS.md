# robocasa_long_horizon Instructions

Write outputs under `runs/autorobobench/robocasa_long_horizon/<run>/`. Do not
edit eval files or split files for scored runs.

## Task

- Optimize one policy for `PickPlaceCounterToMicrowave`.
- Metric: final success plus subgoal progress.
- Default eval: 50 episodes/task, max 750 steps, commit 8.

## Train

```bash
python3 tasks/robocasa_long_horizon/train.py \
  --manifest data/autorobobench/robocasa_long_horizon_manifest.json \
  --split data/autorobobench/robocasa_long_horizon_splits.json \
  --out-dir runs/autorobobench/robocasa_long_horizon/<run> \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_long_horizon/eval.py \
  --checkpoint runs/autorobobench/robocasa_long_horizon/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_long_horizon/<run>/eval_50_per_task.json \
  --eval-episodes-per-task 50 \
  --device cuda
```

## Visualize

Summarize eval/training outputs under `<run>/visualize/`. Add `--render` to also save eval videos.

```bash
python3 tasks/robocasa_long_horizon/visualize.py \
  --run-dir runs/autorobobench/robocasa_long_horizon/<run>
```
