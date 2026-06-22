# robocasa_offlinerl_posttraining Instructions

Write outputs under `runs/autorobobench/robocasa_offlinerl_posttraining/<run>/`. Do not
edit eval files or split files for scored runs.

## Task

- Optimize `PickPlaceCounterToStandMixer` from demonstrations plus offline
  experience: failed rollouts, corrections, or other saved rollouts.
- Metric: rollout success.
- Do not use test-time demos.
- Current measured result: 0/100.

## Train

```bash
python3 tasks/robocasa_offlinerl_posttraining/train.py \
  --manifest data/autorobobench/robocasa_stand_mixer_peak_manifest.json \
  --split data/autorobobench/robocasa_stand_mixer_peak_splits.json \
  --out-dir runs/autorobobench/robocasa_offlinerl_posttraining/<run> \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_offlinerl_posttraining/eval.py \
  --checkpoint runs/autorobobench/robocasa_offlinerl_posttraining/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_offlinerl_posttraining/<run>/eval_10.json \
  --eval-episodes-per-task 10 \
  --device cuda
```

## Visualize

Summarize offline-RL data composition, assigned rewards/advantages, training metrics, and eval success under `<run>/visualize/`.

```bash
python3 tasks/robocasa_offlinerl_posttraining/visualize.py \
  --run-dir runs/autorobobench/robocasa_offlinerl_posttraining/<run>
```
