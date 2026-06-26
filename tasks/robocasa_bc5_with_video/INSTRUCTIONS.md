# robocasa_bc5_with_video Instructions

Write outputs under `runs/autorobobench/robocasa_bc5_with_video/<run>/`. Do not
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

- Train one policy from scarce paired action demos plus RGB-only video.
- Tasks use BC5 task set.
- Data: five paired action demos/task plus video-only pool. The default trainer
  uses PIDM-style auxiliary pretraining: paired demos supervise inverse
  dynamics while RGB-only clips supervise future visual latent prediction.
- Metric: rollout success rate over the five tasks plus a small held-out
  imitation term. Final score is
  `0.95 * success_rate + 0.05 * val_action_mse_score`, where
  `val_action_mse_score = clamp(1 - MSE, 0, 1)` on frozen validation
  trajectories. Paired-action efficiency and data-budget integrity are reported
  as diagnostics/constraints, not score terms.
- Default eval: 100 total rollouts: 20 episodes/task over five tasks, max 260
  steps, commit 16.
- Current smoke evals are 0/100.
- You do not have to use the whole paired or video-only dataset. It is allowed
  to prioritize particular tasks, subsets, or curricula if that improves total
  eval success. Once those tasks are saturated, adding the remaining tasks is
  necessary to approach 100% overall score.
- Test-time inference may not read manifests, splits, datasets, video pools, or
  replay stored trajectories. `inference.py` may use only checkpoint
  weights/statistics plus the current `obs` and `task`.

## Things To Try

- Pretrain visual features or latent dynamics from RGB-only video before
  behavior cloning.
- Train inverse dynamics on video frames, then behavior-clone inferred action
  trajectories where confidence is high.
- Predict future visual latents from video clips.
- Condition the policy on goal or subgoal frames.
- Align scarce robot demos with similar video clips using contrastive losses.
- Use task/subset curricula from the scarce paired demos before expanding.
- Tune training throughput and stability: batch size, gradient accumulation,
  mixed precision (`bf16`, `fp16`, or `fp32`), `torch.compile`, dataloader
  workers, caching/precompute, and checkpoint/eval cadence.

## Train

```bash
python3 tasks/robocasa_bc5_with_video/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_with_video_splits.json \
  --video-pool data/autorobobench/robocasa_bc5_with_video_video_pool.json \
  --out-dir runs/autorobobench/robocasa_bc5_with_video/<run> \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_bc5_with_video/eval_parallel.py \
  --checkpoint runs/autorobobench/robocasa_bc5_with_video/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_bc5_with_video/<run>/eval_100_total.json \
  --eval-episodes-per-task 20 \
  --workers 28 \
  --device cuda
```

## Visualize

Summarize eval/training outputs under `<run>/visualize/`. Add `--render` to also save eval videos.

```bash
python3 tasks/robocasa_bc5_with_video/visualize.py \
  --run-dir runs/autorobobench/robocasa_bc5_with_video/<run>
```
