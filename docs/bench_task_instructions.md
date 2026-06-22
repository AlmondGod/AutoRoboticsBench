# Bench Task Instructions

Use this as the operating checklist for the current bench tasks. Keep runs under
the task training cap unless a task explicitly says otherwise. Write outputs
under `runs/autorobobench/<task>/<run_name>/`. Do not edit eval files or split
files for scored runs.

## Common Commands

Hash immutable benchmark files:

```bash
python3 setup.py --hash-manifest --suite autorobobench_v0 --out runs/autorobobench/v0_hashes.json
```

Parallel RoboCasa rollout eval:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH=third_party/robocasa:third_party/robosuite:$PYTHONPATH \
python3 tasks/robocasa_bc5/eval_parallel.py \
  --manifest <manifest.json> \
  --split <split.json> \
  --inference <python.module.inference> \
  --checkpoint <policy.pt> \
  --out <eval.json> \
  --eval-episodes-per-task 10 \
  --max-steps <steps> \
  --commit-steps <steps> \
  --workers <n> \
  --device cuda
```

Single-process RoboCasa render:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH=third_party/robocasa:third_party/robosuite:$PYTHONPATH \
python3 tasks/robocasa_bc5/eval.py \
  --manifest <manifest.json> \
  --split <split.json> \
  --inference <python.module.inference> \
  --checkpoint <policy.pt> \
  --out <eval_render.json> \
  --eval-episodes-per-task 1 \
  --render-dir <render_dir> \
  --render-episodes-per-task 1 \
  --device cuda
```

Task artifact visualization:

```bash
python3 tasks/<task>/visualize.py --run-dir runs/autorobobench/<task>/<run>
```

## `robocasa_bc5`

- Optimize one policy for `OpenCabinet`, `CloseDrawer`, `CloseFridge`,
  `TurnOffStove`, `PickPlaceCounterToCabinet`.
- Metric: rollout success rate over the five tasks.
- Default eval: 10 episodes/task, max 260 steps, commit 16 unless checkpoint
  overrides commit horizon.
- Data: use `data/robocasa5/manifest.json` and
  `data/autorobobench/robocasa_bc5_splits.json`.
- Train:

```bash
python3 tasks/robocasa_bc5/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_splits.json \
  --out-dir runs/autorobobench/robocasa_bc5/<run> \
  --max-train-seconds 300 \
  --device cuda
```

- Eval:

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

## `robocasa_bc1`

- Optimize one policy for `TurnOnSinkFaucet`.
- Metric: single-task reliability. Report success out of 100.
- Training cap: 300 seconds.
- Data: task-specific trajectories are allowed. Generic video-only pool is
  allowed for training only.
- Test-time replay is banned. `inference.py` and the submitted checkpoint may
  not read or carry demonstration trajectories, trajectory banks, manifest/split
  files, datasets, video pools, or per-episode action arrays during eval. The
  only eval-time inputs are the checkpoint's learned weights/statistics, `obs`,
  and `task`.
- Current learned base: `robocasa_faucet_direct_bc_all_data_5min_seed0`,
  6/10 success, reported as 60/100 normalized.
- Train:

```bash
python3 tasks/robocasa_bc5/train.py \
  --manifest data/autorobobench/robocasa_bc1_manifest.json \
  --split data/autorobobench/robocasa_bc1_splits.json \
  --out-dir runs/autorobobench/robocasa_bc1/<run> \
  --policy-kind bc \
  --chunk-horizon 32 \
  --progress-conditioning \
  --progress-scale 750 \
  --eval-commit-steps 8 \
  --max-train-seconds 300 \
  --device cuda
```

- Eval:

```bash
python3 tasks/robocasa_bc5/eval_parallel.py \
  --manifest data/autorobobench/robocasa_bc1_manifest.json \
  --split data/autorobobench/robocasa_bc1_splits.json \
  --inference tasks.robocasa_bc1.inference \
  --checkpoint runs/autorobobench/robocasa_bc1/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_bc1/<run>/eval_10.json \
  --eval-episodes-per-task 10 \
  --max-steps 400 \
  --commit-steps 8 \
  --workers 5 \
  --device cuda
```

## `robocasa_offlinerl_posttraining`

- Optimize `PickPlaceCounterToStandMixer` from demonstrations plus offline
  experience: failed rollouts, corrections, or other saved rollouts.
- Metric: rollout success.
- Do not use test-time demos.
- Current measured result: 0/100.
- Train:

```bash
python3 tasks/robocasa_offlinerl_posttraining/train.py \
  --manifest data/autorobobench/robocasa_stand_mixer_peak_manifest.json \
  --split data/autorobobench/robocasa_stand_mixer_peak_splits.json \
  --out-dir runs/autorobobench/robocasa_offlinerl_posttraining/<run> \
  --device cuda
```

- Eval:

```bash
python3 tasks/robocasa_offlinerl_posttraining/eval.py \
  --checkpoint runs/autorobobench/robocasa_offlinerl_posttraining/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_offlinerl_posttraining/<run>/eval_10.json \
  --eval-episodes-per-task 10 \
  --device cuda
```

## `robocasa_language_following`

- Optimize one language-conditioned policy over four variants:
  `ChooseMeasuringCupLeftLarger`, `ChooseMeasuringCupLeftSmaller`,
  `ChooseMeasuringCupRightLarger`, `ChooseMeasuringCupRightSmaller`.
- Metric: language-conditioned success. Also report wrong-language success and
  conditioning gap.
- Do not collapse variants into one unlabeled task.
- Train:

```bash
python3 tasks/robocasa_language_following/train.py \
  --manifest data/autorobobench/robocasa_language_following_manifest.json \
  --split data/autorobobench/robocasa_language_following_splits.json \
  --out-dir runs/autorobobench/robocasa_language_following/<run> \
  --device cuda
```

- Eval:

```bash
python3 tasks/robocasa_language_following/eval.py \
  --checkpoint runs/autorobobench/robocasa_language_following/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_language_following/<run>/eval.json \
  --device cuda
```

## `robocasa_long_horizon`

- Optimize one policy for `PickPlaceCounterToMicrowave`.
- Metric: final success plus subgoal progress.
- Default eval: 10 episodes/task, max 750 steps, commit 8.
- Train:

```bash
python3 tasks/robocasa_long_horizon/train.py \
  --manifest data/autorobobench/robocasa_long_horizon_manifest.json \
  --split data/autorobobench/robocasa_long_horizon_splits.json \
  --out-dir runs/autorobobench/robocasa_long_horizon/<run> \
  --device cuda
```

- Eval:

```bash
python3 tasks/robocasa_long_horizon/eval.py \
  --checkpoint runs/autorobobench/robocasa_long_horizon/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_long_horizon/<run>/eval_10_per_task.json \
  --device cuda
```

## `robocasa_world_model`

- Train a state/action world model on BC5 transitions.
- Inputs: `state_t`, `action_t`, `task_id`, progress.
- Targets: next state, next progress, success.
- Metric: policy ranking/calibration against real rollout success plus
  transition prediction metrics.
- This is not a policy rollout score.
- Train:

```bash
python3 tasks/robocasa_world_model/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_splits.json \
  --out-dir runs/autorobobench/robocasa_world_model/<run> \
  --device cuda
```

- Eval:

```bash
python3 tasks/robocasa_world_model/eval.py \
  --checkpoint runs/autorobobench/robocasa_world_model/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_world_model/<run>/eval_correlation.json \
  --device cuda
```

## `robocasa_visual_world_model`

- Train a visual world model on BC5 transitions and videos.
- Inputs: state, action, task, progress, current RGB.
- Targets: next RGB, next state, next progress, success.
- Metric: visual world-model score. LPIPS next-frame quality is the main term.
- This is not a policy rollout score.
- Train:

```bash
python3 tasks/robocasa_visual_world_model/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_splits.json \
  --out-dir runs/autorobobench/robocasa_visual_world_model/<run> \
  --device cuda
```

- Eval:

```bash
python3 tasks/robocasa_visual_world_model/eval.py \
  --checkpoint runs/autorobobench/robocasa_visual_world_model/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_visual_world_model/<run>/eval_lpips.json \
  --device cuda
```

## `robocasa_world_model_posttraining`

- Start from the best differentiable `robocasa_stand_mixer_peak` policy.
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
- Train:

```bash
python3 tasks/robocasa_world_model_posttraining/train.py \
  --world-model-checkpoint <world_model.pt> \
  --out-dir runs/autorobobench/robocasa_world_model_posttraining/<run> \
  --max-train-seconds 300 \
  --device cuda
```

- Eval:

```bash
python3 tasks/robocasa_world_model_posttraining/eval_parallel.py \
  --checkpoint runs/autorobobench/robocasa_world_model_posttraining/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_world_model_posttraining/<run>/eval.json \
  --eval-episodes-per-task 10 \
  --device cuda
```

## `robocasa_bc5_with_video`

- Train one policy from scarce paired action demos plus RGB-only video.
- Tasks use BC5 task set.
- Data: two paired action demos/task plus video-only pool.
- Metric: rollout success and paired-action efficiency.
- Current smoke evals are 0/100.
- Train:

```bash
python3 tasks/robocasa_bc5_with_video/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_with_video_splits.json \
  --video-pool data/autorobobench/robocasa_bc5_with_video_video_pool.json \
  --out-dir runs/autorobobench/robocasa_bc5_with_video/<run> \
  --device cuda
```

- Eval:

```bash
python3 tasks/robocasa_bc5_with_video/eval.py \
  --checkpoint runs/autorobobench/robocasa_bc5_with_video/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_bc5_with_video/<run>/eval.json \
  --device cuda
```
