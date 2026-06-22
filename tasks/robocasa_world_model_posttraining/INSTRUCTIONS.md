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
  `runs/autorobobench/robocasa_stand_mixer_peak/a100_5min_full_seed0/policy_best.pt`.

## Train

```bash
python3 tasks/robocasa_world_model_posttraining/train.py \
  --world-model-checkpoint <world_model.pt> \
  --out-dir runs/autorobobench/robocasa_world_model_posttraining/<run> \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_world_model_posttraining/eval_parallel.py \
  --checkpoint runs/autorobobench/robocasa_world_model_posttraining/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_world_model_posttraining/<run>/eval.json \
  --eval-episodes-per-task 10 \
  --device cuda
```

## Visualize

Summarize world-model posttraining objective, anchor losses, and real eval success under `<run>/visualize/`.

```bash
python3 tasks/robocasa_world_model_posttraining/visualize.py \
  --run-dir runs/autorobobench/robocasa_world_model_posttraining/<run>
```
