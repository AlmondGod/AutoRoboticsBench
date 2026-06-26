# robocasa_long_horizon Instructions

Write outputs under `runs/autorobobench/robocasa_long_horizon/<run>/`. Do not
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

- Optimize one policy for `PickPlaceCounterToMicrowave`.
- Metric: rollout success rate plus a small held-out imitation term. Final
  score is `0.95 * success_rate + 0.05 * val_action_mse_score`, where
  `val_action_mse_score = clamp(1 - MSE, 0, 1)` on frozen validation
  trajectories. Rollout success remains the dominant score component.
- Default eval: 100 rollouts, max 750 steps, commit 8.
- Test-time inference may not read manifests, splits, datasets, video pools, or
  replay stored trajectories. `inference.py` may use only checkpoint
  weights/statistics plus the current `obs` and `task`.

## Things To Try

- Use hierarchical or subgoal policies with waypoint or phase prediction.
- Add longer context with recurrent or transformer policies.
- Train curricula over shorter phases before the full microwave task.
- Condition on goal or subgoal frames.
- Tune commit horizon and recovery behavior after partial failures.
- Warm start from BC1/BC5 encoders or multi-task policies.
- Tune training throughput and stability: batch size, gradient accumulation,
  mixed precision (`bf16`, `fp16`, or `fp32`), `torch.compile`, dataloader
  workers, caching/precompute, and checkpoint/eval cadence.

## Train

```bash
python3 tasks/robocasa_long_horizon/train.py \
  --manifest data/autorobobench/robocasa_long_horizon_manifest.json \
  --split data/autorobobench/robocasa_long_horizon_splits.json \
  --out-dir runs/autorobobench/robocasa_long_horizon/<run> \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_long_horizon/eval.py \
  --checkpoint runs/autorobobench/robocasa_long_horizon/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_long_horizon/<run>/eval_100.json \
  --eval-episodes-per-task 100 \
  --device cuda
```

## Visualize

Summarize eval/training outputs under `<run>/visualize/`. Add `--render` to also save eval videos.

```bash
python3 tasks/robocasa_long_horizon/visualize.py \
  --run-dir runs/autorobobench/robocasa_long_horizon/<run>
```
