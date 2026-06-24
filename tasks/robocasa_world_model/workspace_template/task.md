# robocasa_reward_model Instructions

Write outputs under `runs/autorobobench/robocasa_reward_model/<run>/`. Do not
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

- Train a state/action reward model on BC5 transitions.
- Inputs: `state_t`, `action_t`.
- Targets: next progress and task success/reward only. Do not optimize or score
  next-proprio / next-state prediction.
- `task_id` and current task progress are split metadata / labels only; they
  must not be passed to the reward model as conditioning inputs.
- Metric: policy ranking/calibration against real rollout success plus
  reward/progress prediction metrics.
- This is not a policy rollout score. By default eval scores existing policy
  traces only; pass `--generate-missing-traces` for simulator/offline trace
  materialization.
- Test-time inference may not read train/eval manifests, split files, datasets,
  or video pools. `inference.py` may use only checkpoint weights/statistics plus
  the current state/action inputs supplied by eval.

## Things To Try

- Scale width, depth, latent size, and regularization up or down.
- Calibrate success probabilities and tune BCE versus progress loss weights.
- Add hard negatives, failed rollouts, and out-of-distribution policy traces.
- Use ensembles or uncertainty penalties for policy ranking.
- Try contrastive or ranking losses to improve correlation with real eval
  success.
- Try transformer or residual dynamics heads, different activations, optimizers,
  and schedules.
- Tune training throughput and stability: batch size, gradient accumulation,
  mixed precision (`bf16`, `fp16`, or `fp32`), `torch.compile`, dataloader
  workers, caching/precompute, and checkpoint/eval cadence.

## Train

```bash
python3 tasks/robocasa_world_model/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_splits.json \
  --out-dir runs/autorobobench/robocasa_reward_model/<run> \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_world_model/eval.py \
  --checkpoint runs/autorobobench/robocasa_reward_model/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_reward_model/<run>/eval_correlation.json \
  --device cuda
```

## Visualize

Summarize reward-model policy ranking, calibration, reward/progress metrics, and real-vs-predicted policy scores under `<run>/visualize/`.

```bash
python3 tasks/robocasa_world_model/visualize.py \
  --run-dir runs/autorobobench/robocasa_reward_model/<run>
```
