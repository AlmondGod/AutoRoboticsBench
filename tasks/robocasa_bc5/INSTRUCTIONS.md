# robocasa_bc5 Instructions

Keep scored runs under the task training cap. Write outputs under
`runs/autorobobench/robocasa_bc5/<run>/`. Do not edit eval files or split files
for scored runs.

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

- Optimize one policy for `OpenCabinet`, `CloseDrawer`, `CloseFridge`,
  `TurnOffStove`, `PickPlaceCounterToCabinet`.
- Metric: rollout success rate over the five tasks plus a small held-out
  imitation term. Final score is
  `0.95 * success_rate + 0.05 * val_action_mse_score`, where
  `val_action_mse_score = clamp(1 - MSE, 0, 1)` on frozen validation
  trajectories. Rollout success remains the dominant score component.
- Default eval: 100 total rollouts: 20 episodes/task over five tasks, max 260
  steps, commit 16 unless checkpoint overrides commit horizon.
- Data: use `data/robocasa5/manifest.json` and
  `data/autorobobench/robocasa_bc5_splits.json`.
- You do not have to train on the whole dataset. It is allowed to prioritize
  particular tasks, subsets, or curricula if that improves total eval success.
  Once those tasks are saturated, adding the remaining tasks is necessary to
  approach 100% overall score.
- Test-time inference may not read manifests, splits, datasets, video pools, or
  replay stored trajectories. `inference.py` may use only checkpoint
  weights/statistics plus the current `obs` and `task`.

## Things To Try

- Scale model width, depth, chunk horizon, and image encoder capacity up or
  down.
- Try ACT/transformer/history policies, CNN, ResNet, or ViT backbones, and
  residual MLP action heads.
- Try flow or diffusion action-chunk modeling.
- Tune commit horizon, action smoothing, per-task normalization, and
  task-balanced sampling.
- Prioritize weak tasks or subset curricula before broadening to all five tasks.
- Tune training throughput and stability: batch size, gradient accumulation,
  mixed precision (`bf16`, `fp16`, or `fp32`), `torch.compile`, dataloader
  workers, caching/precompute, and checkpoint/eval cadence.

## Train

```bash
python3 tasks/robocasa_bc5/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_splits.json \
  --out-dir runs/autorobobench/robocasa_bc5/<run> \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_bc5/eval_parallel.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_splits.json \
  --inference tasks.robocasa_bc5.inference \
  --checkpoint runs/autorobobench/robocasa_bc5/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_bc5/<run>/eval_100_total.json \
  --eval-episodes-per-task 20 \
  --max-steps 260 \
  --commit-steps 16 \
  --workers 28 \
  --device cuda
```

## Render

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH=third_party/robocasa:third_party/robosuite:$PYTHONPATH \
python3 tasks/robocasa_bc5/eval.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_splits.json \
  --inference tasks.robocasa_bc5.inference \
  --checkpoint runs/autorobobench/robocasa_bc5/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_bc5/<run>/eval_render.json \
  --eval-episodes-per-task 1 \
  --render-dir runs/autorobobench/robocasa_bc5/<run>/videos \
  --render-episodes-per-task 1 \
  --device cuda
```

## Visualize

Summarize eval/training outputs under `<run>/visualize/`. Add `--render` to also save eval videos.

```bash
python3 tasks/robocasa_bc5/visualize.py \
  --run-dir runs/autorobobench/robocasa_bc5/<run>
```
