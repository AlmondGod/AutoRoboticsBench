# robocasa_bc1 Instructions

Keep scored runs under 300 seconds. Write outputs under
`runs/autorobobench/robocasa_bc1/<run>/`. Do not edit eval files or
split files for scored runs.

## Task

- Optimize one policy for `TurnOnSinkFaucet`.
- Metric: primarily single-task reliability, reported out of 100. Task speed is
  a secondary factor with much lower weight: faster successful eval episodes
  help slightly, but an unsuccessful episode is never better than a successful
  one.
- Data: task-specific trajectories are allowed. Generic video-only pool is
  allowed for training only.
- Test-time replay is banned. `inference.py` and the submitted checkpoint may
  not read or carry demonstration trajectories, trajectory banks, manifest/split
  files, datasets, video pools, or per-episode action arrays during eval. The
  only eval-time inputs are the checkpoint's learned weights/statistics, `obs`,
  and `task`.
- Current learned base: `robocasa_faucet_direct_bc_all_data_5min_seed0`,
  6/10 success, reported as 60/100 normalized.

## Train

```bash
python3 tasks/robocasa_bc5/train.py \
  --manifest data/autorobobench/robocasa_bc1_manifest.json \
  --split data/autorobobench/robocasa_bc1_splits.json \
  --out-dir runs/autorobobench/robocasa_bc1/<run> \
  --policy-kind bc \
  --chunk-horizon 32 \
  --progress-conditioning \
  --progress-scale 750 \
  --eval-commit-steps 8 \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_bc1/eval_parallel.py \
  --checkpoint runs/autorobobench/robocasa_bc1/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_bc1/<run>/eval_50.json \
  --eval-episodes-per-task 50 \
  --max-steps 400 \
  --commit-steps 8 \
  --workers 28 \
  --device cuda
```

## Visualize

Summarize eval/training outputs under `<run>/visualize/`. Add `--render` to also save eval videos.

```bash
python3 tasks/robocasa_bc1/visualize.py \
  --run-dir runs/autorobobench/robocasa_bc1/<run>
```
