# robocasa_bc5 Instructions

Keep scored runs under the task training cap. Write outputs under
`runs/autorobobench/robocasa_bc5/<run>/`. Do not edit eval files or split files
for scored runs.

## Task

- Optimize one policy for `OpenCabinet`, `CloseDrawer`, `CloseFridge`,
  `TurnOffStove`, `PickPlaceCounterToCabinet`.
- Metric: rollout success rate over the five tasks.
- Default eval: 10 episodes/task, max 260 steps, commit 16 unless checkpoint
  overrides commit horizon.
- Data: use `data/robocasa5/manifest.json` and
  `data/autorobobench/robocasa_bc5_splits.json`.
- You do not have to train on the whole dataset. It is allowed to prioritize
  particular tasks, subsets, or curricula if that improves total eval success.
  Once those tasks are saturated, adding the remaining tasks is necessary to
  approach 100% overall score.

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
  --out runs/autorobobench/robocasa_bc5/<run>/eval_10_per_task.json \
  --eval-episodes-per-task 10 \
  --max-steps 260 \
  --commit-steps 16 \
  --workers 10 \
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
