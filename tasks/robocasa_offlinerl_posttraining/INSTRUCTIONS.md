# robocasa_offlinerl_posttraining Instructions

Write outputs under `runs/autorobobench/robocasa_offlinerl_posttraining/<run>/`. Do not
edit eval files or split files for scored runs.

## Iterative Research Protocol

- Do not queue up a batch of experiments to run unattended.
- Think of experiments one by one: state one hypothesis, make the smallest
  relevant change, train or run the check, inspect loss/eval output, then choose
  the next experiment from that result.
- Build experiments cumulatively on the current per-run branch. When a source
  change improves eval score or a task-relevant validation/loss signal, record
  the evidence and commit it before starting the next experiment.
- The next successful experiment should start on top of the previous successful
  committed change. If a change fails or is worse, discard it or explicitly
  supersede it before moving on.
- Never merge an agent run branch to `main`.

## Task

- Optimize `PickPlaceCounterToStandMixer` from demonstrations plus offline
  experience: successful rollouts, failed rollouts, corrections, or other saved
  rollouts.
- Use the pretrained reward/progress model when available to score offline
  transitions and assign advantages for policy improvement.
- Metric: rollout success rate.
- Do not use test-time demos.
- Default warm start path:
  `runs/autorobobench/robocasa_stand_mixer_base/nonzero_base/policy_best.pt`.
- Default reward model path:
  `data/autorobobench/pretrained_reward_models/robocasa_stand_mixer_reward_model.pt`.
- Current promoted A100 base: learned temporal chunk BC, 2/10 eval success on
  `PickPlaceCounterToStandMixer`. It was trained with an eval-included
  diagnostic split to provide a nonzero warm start; do not report it as a fair
  standalone benchmark submission.
- Test-time inference may not read manifests, splits, datasets, video pools, or
  replay stored trajectories. `inference.py` may use only checkpoint
  weights/statistics plus the current `obs` and `task`.

## Things To Try

- Use reward-weighted BC or advantage-weighted regression with the pretrained
  reward model.
- Try conservative Q-learning, IQL, or AWR-style objectives on successful and
  failed trajectories.
- Filter or reweight failed/correction segments by reward-model confidence.
- Clone corrections while keeping a behavior-regularization anchor to the warm
  start policy.
- Tune reward-model advantage weight, bad-rollout ratio, correction fraction,
  and init anchor strength.
- Tune training throughput and stability: batch size, gradient accumulation,
  mixed precision (`bf16`, `fp16`, or `fp32`), `torch.compile`, dataloader
  workers, caching/precompute, and checkpoint/eval cadence.

## Train

```bash
python3 tasks/robocasa_offlinerl_posttraining/train.py \
  --manifest data/autorobobench/robocasa_stand_mixer_peak_manifest.json \
  --split data/autorobobench/robocasa_stand_mixer_peak_splits.json \
  --out-dir runs/autorobobench/robocasa_offlinerl_posttraining/<run> \
  --reward-model-checkpoint data/autorobobench/pretrained_reward_models/robocasa_stand_mixer_reward_model.pt \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_offlinerl_posttraining/eval_parallel.py \
  --checkpoint runs/autorobobench/robocasa_offlinerl_posttraining/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_offlinerl_posttraining/<run>/eval_100.json \
  --eval-episodes-per-task 100 \
  --workers 28 \
  --device cuda
```

## Visualize

Summarize offline-RL data composition, assigned rewards/advantages, training metrics, and eval success under `<run>/visualize/`.

```bash
python3 tasks/robocasa_offlinerl_posttraining/visualize.py \
  --run-dir runs/autorobobench/robocasa_offlinerl_posttraining/<run>
```
