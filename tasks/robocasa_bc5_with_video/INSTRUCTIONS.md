# robocasa_bc5_with_video Instructions

Write outputs under `runs/autorobobench/robocasa_bc5_with_video/<run>/`. Do not
edit eval files or split files for scored runs.

## Task

- Train one policy from scarce paired action demos plus RGB-only video.
- Tasks use BC5 task set.
- Data: five paired action demos/task plus video-only pool. The default trainer
  uses PIDM-style auxiliary pretraining: paired demos supervise inverse
  dynamics while RGB-only clips supervise future visual latent prediction.
- Metric: rollout success and paired-action efficiency.
- Current smoke evals are 0/100.
- You do not have to use the whole paired or video-only dataset. It is allowed
  to prioritize particular tasks, subsets, or curricula if that improves total
  eval success. Once those tasks are saturated, adding the remaining tasks is
  necessary to approach 100% overall score.

## Train

```bash
python3 tasks/robocasa_bc5_with_video/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_with_video_splits.json \
  --video-pool data/autorobobench/robocasa_bc5_with_video_video_pool.json \
  --out-dir runs/autorobobench/robocasa_bc5_with_video/<run> \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_bc5_with_video/eval.py \
  --checkpoint runs/autorobobench/robocasa_bc5_with_video/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_bc5_with_video/<run>/eval.json \
  --device cuda
```

## Visualize

Summarize eval/training outputs under `<run>/visualize/`. Add `--render` to also save eval videos.

```bash
python3 tasks/robocasa_bc5_with_video/visualize.py \
  --run-dir runs/autorobobench/robocasa_bc5_with_video/<run>
```
