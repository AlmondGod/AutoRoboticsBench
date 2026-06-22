# robocasa_offlinerl_posttraining Instructions

Write outputs under `runs/autorobobench/robocasa_offlinerl_posttraining/<run>/`. Do not
edit eval files or split files for scored runs.

## Task

- Optimize `PickPlaceCounterToStandMixer` from demonstrations plus offline
  experience: successful rollouts, failed rollouts, corrections, or other saved
  rollouts.
- Use the pretrained reward/progress model when available to score offline
  transitions and assign advantages for policy improvement.
- Metric: rollout success.
- Do not use test-time demos.
- Default warm start path:
  `runs/autorobobench/robocasa_stand_mixer_base/nonzero_base/policy_best.pt`.
- Default reward model path:
  `data/autorobobench/pretrained_reward_models/robocasa_stand_mixer_reward_model.pt`.
- Current promoted A100 base: learned temporal chunk BC, 2/10 eval success on
  `PickPlaceCounterToStandMixer`. It was trained with an eval-included
  diagnostic split to provide a nonzero warm start; do not report it as a fair
  standalone benchmark submission.

## Train

```bash
python3 tasks/robocasa_offlinerl_posttraining/train.py \
  --manifest data/autorobobench/robocasa_stand_mixer_peak_manifest.json \
  --split data/autorobobench/robocasa_stand_mixer_peak_splits.json \
  --out-dir runs/autorobobench/robocasa_offlinerl_posttraining/<run> \
  --reward-model-checkpoint data/autorobobench/pretrained_reward_models/robocasa_stand_mixer_reward_model.pt \
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
