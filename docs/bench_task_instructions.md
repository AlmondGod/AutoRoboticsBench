# Bench Task Instructions

Use this as the operating checklist for the current bench tasks. Keep runs under
the task training cap unless a task explicitly says otherwise. Write outputs
under `runs/autorobobench/<task>/<run_name>/`. Do not edit eval files or split
files for scored runs.

## Common Commands

Verify checked-in policy artifacts:

```bash
python3 -m autorobobench.policy_artifacts --verify
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

## `robocasa_bc5`

- Optimize one policy for `OpenDrawer`, `CloseDrawer`, `CloseFridge`,
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

## `robocasa_faucet_peak`

- Optimize one policy for `TurnOnSinkFaucet`.
- Metric: single-task reliability. Report success out of 100.
- Training cap: 300 seconds.
- Data: task-specific trajectories are allowed. Generic video-only pool is
  allowed. Test-time demo replay is not allowed for learned-policy claims.
- Current learned base: `robocasa_faucet_direct_bc_all_data_5min_seed0`,
  6/10 success, reported as 60/100 normalized.
- Train:

```bash
python3 tasks/robocasa_bc5/train.py \
  --manifest data/autorobobench/robocasa_faucet_peak_manifest.json \
  --split data/autorobobench/robocasa_faucet_peak_splits.json \
  --out-dir runs/autorobobench/robocasa_faucet_peak/<run> \
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
  --manifest data/autorobobench/robocasa_faucet_peak_manifest.json \
  --split data/autorobobench/robocasa_faucet_peak_splits.json \
  --inference tasks.robocasa_bc5.inference \
  --checkpoint runs/autorobobench/robocasa_faucet_peak/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_faucet_peak/<run>/eval_10.json \
  --eval-episodes-per-task 10 \
  --max-steps 400 \
  --commit-steps 8 \
  --workers 5 \
  --device cuda
```

## `robocasa_stand_mixer_peak`

- Optimize one policy for `PickPlaceCounterToStandMixer`.
- Metric: single-task rollout success.
- Training cap: 300 seconds.
- Data: task-specific action demos and generic video-only pool are allowed.
- Current measured learned policies are 0/100.
- Train:

```bash
python3 tasks/robocasa_stand_mixer_peak/train.py \
  --manifest data/autorobobench/robocasa_stand_mixer_peak_manifest.json \
  --split data/autorobobench/robocasa_stand_mixer_peak_splits.json \
  --out-dir runs/autorobobench/robocasa_stand_mixer_peak/<run> \
  --max-train-seconds 300 \
  --device cuda
```

- Eval:

```bash
python3 tasks/robocasa_stand_mixer_peak/eval.py \
  --checkpoint runs/autorobobench/robocasa_stand_mixer_peak/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_stand_mixer_peak/<run>/eval_10.json \
  --eval-episodes-per-task 10 \
  --device cuda
```

## `robocasa_recap_offline`

- Optimize `PickPlaceCounterToStandMixer` from demonstrations plus offline
  experience: failed rollouts, corrections, or other saved rollouts.
- Metric: rollout success.
- Do not use test-time demos.
- Current measured result: 0/100.
- Train:

```bash
python3 tasks/robocasa_recap_offline/train.py \
  --manifest data/autorobobench/robocasa_stand_mixer_peak_manifest.json \
  --split data/autorobobench/robocasa_stand_mixer_peak_splits.json \
  --out-dir runs/autorobobench/robocasa_recap_offline/<run> \
  --device cuda
```

- Eval:

```bash
python3 tasks/robocasa_recap_offline/eval.py \
  --checkpoint runs/autorobobench/robocasa_recap_offline/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_recap_offline/<run>/eval_10.json \
  --eval-episodes-per-task 10 \
  --device cuda
```

## `robocasa_choose_measuring_cup_language`

- Optimize one language-conditioned policy over four variants:
  `ChooseMeasuringCupLeftLarger`, `ChooseMeasuringCupLeftSmaller`,
  `ChooseMeasuringCupRightLarger`, `ChooseMeasuringCupRightSmaller`.
- Metric: language-conditioned success. Also report wrong-language success and
  conditioning gap.
- Do not collapse variants into one unlabeled task.
- Train:

```bash
python3 tasks/robocasa_choose_measuring_cup_language/train.py \
  --manifest data/autorobobench/robocasa_choose_measuring_cup_language_manifest.json \
  --split data/autorobobench/robocasa_choose_measuring_cup_language_splits.json \
  --out-dir runs/autorobobench/robocasa_choose_measuring_cup_language/<run> \
  --device cuda
```

- Eval:

```bash
python3 tasks/robocasa_choose_measuring_cup_language/eval.py \
  --checkpoint runs/autorobobench/robocasa_choose_measuring_cup_language/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_choose_measuring_cup_language/<run>/eval.json \
  --device cuda
```

## `robocasa_long_horizon`

- Optimize one policy for sequential tasks:
  `PickPlaceCounterToStove`, `TurnOffStove`, `PickPlaceCounterToCabinet`.
- Metric: final success plus subgoal progress.
- Default eval: 10 episodes/task, max 750 steps, commit 8.
- Train:

```bash
python3 tasks/robocasa_long_horizon/train.py \
  --manifest data/robocasa5/manifest.json \
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
- Targets: next state, next progress, reward, success.
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
- Targets: next RGB, next state, next progress, reward, success.
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

## `robocasa_wm_policy_improvement`

- Start from a differentiable BC5-compatible policy.
- Use a frozen world model to improve the policy offline.
- Keep BC loss, init-policy anchor, and action penalty active. Real simulator
  success is final; WM objective alone is not enough.
- Supported policy modes: temporal chunk BC, temporal chunk flow, sequence flow.
- Unsupported for v0: trajectory banks, history policies, frozen VLM feature
  cache policies.
- Current checked-in WM-compatible base:
  `data/autorobobench/pretrained_policies/robocasa_faucet_direct_bc_all_data_5min_seed0.pt`.
- Train:

```bash
python3 tasks/robocasa_wm_policy_improvement/train.py \
  --manifest <manifest.json> \
  --split <split.json> \
  --policy-checkpoint <base_policy.pt> \
  --world-model-checkpoint <world_model.pt> \
  --out-dir runs/autorobobench/robocasa_wm_policy_improvement/<run> \
  --max-train-seconds 300 \
  --device cuda
```

- Eval:

```bash
python3 tasks/robocasa_bc5/eval_parallel.py \
  --manifest <manifest.json> \
  --split <split.json> \
  --inference tasks.robocasa_bc5.inference \
  --checkpoint runs/autorobobench/robocasa_wm_policy_improvement/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_wm_policy_improvement/<run>/eval.json \
  --eval-episodes-per-task 10 \
  --workers 5 \
  --device cuda
```

## `robocasa_progress_predictor`

- Train an auxiliary progress regressor on BC5 state/action/task data.
- Metric: validation R2, MAE, RMSE, baseline RMSE.
- This is not a rollout-success benchmark.
- Train:

```bash
python3 -m tasks.robocasa_progress_predictor.train \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_splits.json \
  --out-dir runs/autorobobench/robocasa_progress_predictor/<run> \
  --max-train-seconds 300 \
  --device cuda
```

## `video_policy_transfer`

- Train one policy from scarce paired action demos plus RGB-only video.
- Tasks use BC5 task set.
- Data: two paired action demos/task plus video-only pool.
- Metric: rollout success and paired-action efficiency.
- Current smoke evals are 0/100.
- Train:

```bash
python3 tasks/video_policy_transfer/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/video_policy_transfer_splits.json \
  --video-pool data/autorobobench/video_policy_transfer_video_pool.json \
  --out-dir runs/autorobobench/video_policy_transfer/<run> \
  --device cuda
```

- Eval:

```bash
python3 tasks/video_policy_transfer/eval.py \
  --checkpoint runs/autorobobench/video_policy_transfer/<run>/policy_best.pt \
  --out runs/autorobobench/video_policy_transfer/<run>/eval.json \
  --device cuda
```

## `robotwin_bc3`

- Train one RoboTwin policy for `blocks_ranking_rgb`, `place_a2b_left`,
  `place_object_basket`.
- This is not RoboCasa. Use RoboTwin data and eval scripts.
- Training cap: 300 seconds.
- Metric: success rate for online eval; heldout action MSE for offline eval.
- Train:

```bash
python3 tasks/robotwin_bc3/train.py \
  --split data/autorobobench/robotwin_bc3_splits.json \
  --output-dir runs/autorobobench/robotwin_bc3/<run> \
  --device cuda
```

- Eval:

```bash
python3 tasks/robotwin_bc3/eval.py \
  --checkpoint runs/autorobobench/robotwin_bc3/<run>/policy_best.pt \
  --out runs/autorobobench/robotwin_bc3/<run>/eval.json \
  --device cuda
```
