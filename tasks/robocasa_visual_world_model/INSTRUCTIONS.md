# robocasa_visual_world_model Instructions

Write outputs under `runs/autorobobench/robocasa_visual_world_model/<run>/`.
Do not edit eval files or split files for scored runs.

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

- Train a visual world model on BC5 transitions and videos.
- Inputs: state, action, current RGB.
- Targets: next RGB, next state, next progress, success.
- `task_id` and current task progress are split metadata / labels only; they
  must not be passed to the world model as conditioning inputs.
- Metric: visual world-model score. The main term is correlation between
  closed-loop world-model policy scores and stored real simulator eval success
  for a fixed policy set. Pixel/state/progress/success prediction terms are
  secondary; LPIPS is reported as a diagnostic only.
- This is not a policy rollout score.
- Test-time inference may not read train/eval manifests, split files, datasets,
  or video pools. `inference.py` may use only checkpoint weights/statistics plus
  the current transition inputs supplied by eval.

## Things To Try

- Scale model width, depth, latent size, and image resolution up or down.
- Try L1, Huber, perceptual, SSIM, frequency-weighted, or balanced RGB/state
  reconstruction losses.
- Try different VAE or autoencoder structures: spatial latents, hierarchical
  latents, residual decoders, or stronger encoders.
- Try diffusion or flow matching for next visual latent or next state delta.
- Try transformer, CNN, or residual dynamics heads with different activations,
  optimizers, and augmentations.
- Improve fixed-policy correlation by stabilizing closed-loop generated RGB and
  predicted state.
- Tune training throughput and stability: batch size, gradient accumulation,
  mixed precision (`bf16`, `fp16`, or `fp32`), `torch.compile`, dataloader
  workers, caching/precompute, and checkpoint/eval cadence.

## Train

```bash
python3 tasks/robocasa_visual_world_model/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_splits.json \
  --out-dir runs/autorobobench/robocasa_visual_world_model/<run> \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_visual_world_model/eval.py \
  --checkpoint runs/autorobobench/robocasa_visual_world_model/<run>/policy_best.pt \
  --policy-set data/autorobobench/robocasa_world_model_policy_set.json \
  --out runs/autorobobench/robocasa_visual_world_model/<run>/eval_correlation.json \
  --device cuda
```

## Visualize

Summarize visual world-model metrics under `<run>/visualize/`. Add `--rollout` to save a predicted-vs-actual rollout GIF when a checkpoint is available.

```bash
python3 tasks/robocasa_visual_world_model/visualize.py \
  --run-dir runs/autorobobench/robocasa_visual_world_model/<run>
```
