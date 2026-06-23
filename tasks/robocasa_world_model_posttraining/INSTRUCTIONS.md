# robocasa_world_model_posttraining Instructions

Keep scored runs under 300 seconds. Write outputs under
`runs/autorobobench/robocasa_world_model_posttraining/<run>/`. Do not edit eval
files or split files for scored runs.

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

- Start from a differentiable BC5-compatible policy.
- Use a frozen world model to improve the policy offline.
- Keep BC loss, init-policy anchor, and action penalty active. Real simulator
  success is final; WM objective alone is not enough.
- Supported policy modes: temporal chunk BC, temporal chunk flow, sequence flow.
- Unsupported for v0: trajectory banks, history policies, frozen VLM feature
  cache policies.
- Default task: `PickPlaceCounterToStandMixer` via
  `data/autorobobench/robocasa_stand_mixer_peak_manifest.json` and
  `data/autorobobench/robocasa_stand_mixer_peak_splits.json`.
- Default input policy path:
  `runs/autorobobench/robocasa_stand_mixer_base/nonzero_base/policy_best.pt`.
- Default frozen world-model path:
  `data/autorobobench/pretrained_world_models/robocasa_visual_world_model_spatial_conv_11task_20min.pt`.
- Current promoted world model: VisualRoboCasaWorldModel with spatial VAE
  latents and a conv residual latent-map dynamics head. It was trained on the
  11-task transition suite and reached best validation visual score loss
  `0.0062508`.
- Current promoted A100 base: learned temporal chunk BC, 2/10 eval success on
  `PickPlaceCounterToStandMixer`. It was trained with an eval-included
  diagnostic split to provide a nonzero warm start; do not report it as a fair
  standalone benchmark submission.
- Test-time inference may not read manifests, splits, datasets, video pools, or
  replay stored trajectories. `inference.py` may use only checkpoint
  weights/statistics plus the current `obs` and `task`.

## Things To Try

- Use Dreamer-style latent imagination with value and actor heads.
- Use the frozen or finetuned world model as a policy backbone.
- Generate synthetic trajectories and train only on high predicted reward,
  low-uncertainty ones.
- Use MPC or CEM planning through the world model.
- Rerank policy action chunks by predicted world-model success.
- Add world-model uncertainty penalties or ensemble-disagreement filters.
- Tune training throughput and stability: batch size, gradient accumulation,
  mixed precision (`bf16`, `fp16`, or `fp32`), `torch.compile`, dataloader
  workers, caching/precompute, and checkpoint/eval cadence.

## Train

```bash
python3 tasks/robocasa_world_model_posttraining/train.py \
  --out-dir runs/autorobobench/robocasa_world_model_posttraining/<run> \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_world_model_posttraining/eval_parallel.py \
  --checkpoint runs/autorobobench/robocasa_world_model_posttraining/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_world_model_posttraining/<run>/eval.json \
  --eval-episodes-per-task 100 \
  --device cuda
```

## Visualize

Summarize world-model posttraining objective, anchor losses, and real eval success under `<run>/visualize/`.

```bash
python3 tasks/robocasa_world_model_posttraining/visualize.py \
  --run-dir runs/autorobobench/robocasa_world_model_posttraining/<run>
```
