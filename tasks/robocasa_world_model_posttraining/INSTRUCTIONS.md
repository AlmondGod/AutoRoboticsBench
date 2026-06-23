# robocasa_world_model_posttraining Instructions

Keep scored runs under 300 seconds. Write outputs under
`runs/autorobobench/robocasa_world_model_posttraining/<run>/`. Do not edit eval
files or split files for scored runs.

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
  --eval-episodes-per-task 50 \
  --device cuda
```

## Visualize

Summarize world-model posttraining objective, anchor losses, and real eval success under `<run>/visualize/`.

```bash
python3 tasks/robocasa_world_model_posttraining/visualize.py \
  --run-dir runs/autorobobench/robocasa_world_model_posttraining/<run>
```
