# RoboCasa Task Baselines

This document records the RoboCasa tasks currently wired into the repo and the
best measured base rollout rates we have for each bench. Rates are normalized to
successes per 100 eval episodes. Unless noted otherwise, results came from the
A100 runtime at `/workspace/robot-autoresearch`.

## Code Cleanliness

Current repo state after the single-task policy work:

- The single-task policy code and checked-in artifacts are committed and pushed.
- The A100 runtime copy was synced from the pushed commit for the changed code
  and checked-in policy artifacts.
- `python3 -m autorobobench.policy_artifacts --verify` passes for checked-in
  policy artifacts.
- `py_compile` passes for the touched Python modules:
  `tasks/robocasa_bc5/inference.py`,
  `tasks/robocasa_bc5/eval_parallel.py`,
  `tasks/robocasa_bc5/build_trajectory_bank.py`,
  `tasks/robocasa_progress_predictor/train.py`,
  `tasks/robocasa_wm_policy_improvement/selector_inference.py`, and
  `autorobobench/policy_artifacts.py`.

## Configured Bench Tasks

| Bench task | Current RoboCasa task(s) | Manifest | Split |
|---|---|---|---|
| `robocasa_bc5` | `OpenDrawer`, `CloseDrawer`, `CloseFridge`, `TurnOffStove`, `PickPlaceCounterToCabinet` | `data/robocasa5/manifest.json` | `data/autorobobench/robocasa_bc5_splits.json` |
| `robocasa_world_model` | same as `robocasa_bc5` | `data/robocasa5/manifest.json` | `data/autorobobench/robocasa_bc5_splits.json` |
| `robocasa_visual_world_model` | same as `robocasa_bc5` | `data/robocasa5/manifest.json` | `data/autorobobench/robocasa_bc5_splits.json` |
| `robocasa_wm_policy_improvement` | defaults to same as `robocasa_bc5`; recent focused run used `TurnOnSinkFaucet` | `data/robocasa5/manifest.json` by default | `data/autorobobench/robocasa_bc5_splits.json` by default |
| `robocasa_progress_predictor` | same as `robocasa_bc5` | `data/robocasa5/manifest.json` | `data/autorobobench/robocasa_bc5_splits.json` |
| `robocasa_faucet_peak` | `TurnOnSinkFaucet` | `data/autorobobench/robocasa_faucet_peak_manifest.json` | `data/autorobobench/robocasa_faucet_peak_splits.json` |
| `robocasa_stand_mixer_peak` | `PickPlaceCounterToStandMixer` | `data/autorobobench/robocasa_stand_mixer_peak_manifest.json` | `data/autorobobench/robocasa_stand_mixer_peak_splits.json` |
| `robocasa_recap_offline` | `PickPlaceCounterToStandMixer` | `data/autorobobench/robocasa_stand_mixer_peak_manifest.json` | `data/autorobobench/robocasa_stand_mixer_peak_splits.json` |
| `robocasa_choose_measuring_cup_language` | `ChooseMeasuringCupLeftLarger`, `ChooseMeasuringCupLeftSmaller`, `ChooseMeasuringCupRightLarger`, `ChooseMeasuringCupRightSmaller` | `data/autorobobench/robocasa_choose_measuring_cup_language_manifest.json` | `data/autorobobench/robocasa_choose_measuring_cup_language_splits.json` |
| `robocasa_long_horizon` | `PickPlaceCounterToStove`, `TurnOffStove`, `PickPlaceCounterToCabinet` | `data/robocasa5/manifest.json` | `data/autorobobench/robocasa_long_horizon_splits.json` |

Additional single-task/debug splits currently exist for `CloseFridge`,
`CloseCabinet`, and same-setting `TurnOnSinkFaucet`.

## Base Rollout Rates

| Scope | Policy / method | Type | Eval episodes | Successes | Rate out of 100 | Source |
|---|---|---:|---:|---:|---:|---|
| `robocasa_faucet_peak` / `TurnOnSinkFaucet` | `robocasa_faucet_direct_bc_all_data_5min_seed0` | learned BC | 10 | 6 | 60 | `data/autorobobench/pretrained_policy_evals/robocasa_faucet_direct_bc_all_data_5min_seed0_eval_source_10x5.json` |
| `robocasa_faucet_peak` / `TurnOnSinkFaucet` | `faucet_direct_bc_wm_5min` | learned BC + WM objective | 10 | 4 | 40 | `data/autorobobench/pretrained_policy_evals/robocasa_faucet_direct_bc_wm_aggressive_eval_10x5.json` |
| `robocasa_faucet_peak` / `TurnOnSinkFaucet` | `faucet_direct_bc_wm_conservative_5min` | learned BC + conservative WM objective | 10 | 6 | 60 | `data/autorobobench/pretrained_policy_evals/robocasa_faucet_direct_bc_wm_conservative_eval_10x5.json` |
| `robocasa_faucet_peak` / `TurnOnSinkFaucet` | `trajectory_bank_all107_rgb16` | replay bank | 10 | 10 | 100 | `runs/autorobobench/robocasa_faucet_peak/trajectory_bank_all107_rgb16/eval_10.json` |
| `robocasa_faucet_peak` / `TurnOnSinkFaucet` | `trajectory_bank_all107_rgb16` | replay bank, visual selection | 10 | 9 | 90 | `runs/autorobobench/robocasa_faucet_peak/trajectory_bank_all107_rgb16/eval_10_visual.json` |
| `robocasa_faucet_peak` / `TurnOnSinkFaucet` | `trajectory_bank_all107_rgb16` | replay bank, anti-replay starts | 10 | 6 | 60 | `runs/autorobobench/robocasa_faucet_peak/trajectory_bank_all107_rgb16/eval_10_visual_antireplay.json` |
| `robocasa_close_fridge_full_dataset` / `CloseFridge` | `robocasa_close_fridge_trajectory_bank_all106_rgb16` | replay bank | 32 | 32 | 100 | `data/autorobobench/pretrained_policy_evals/robocasa_close_fridge_trajectory_bank_all106_rgb16_eval_32x16.json` |
| `robocasa_close_fridge_full_dataset` / `CloseFridge` | `history_act_seed0_5min` | learned history ACT | 8 | 0 | 0 | `runs/autorobobench/robocasa_close_fridge_full_dataset/history_act_seed0_5min/eval_parallel_smoke_8x4.json` |
| `robocasa_close_cabinet_peak` / `CloseCabinet` | `five_min_history_act_seed0` | learned history ACT | 4 | 0 | 0 | `runs/autorobobench/robocasa_close_cabinet_peak/five_min_history_act_seed0/eval_4_max400.json` |
| `robocasa_close_cabinet_peak` / `CloseCabinet` | `five_min_frozen_clip_seed0` | learned frozen CLIP flow | 4 | 0 | 0 | `runs/autorobobench/robocasa_close_cabinet_peak/five_min_frozen_clip_seed0/eval_4_max400.json` |
| `robocasa_stand_mixer_peak` / `PickPlaceCounterToStandMixer` | `a100_5min_full_seed0` | learned BC | 10 | 0 | 0 | `runs/autorobobench/robocasa_stand_mixer_peak/a100_5min_full_seed0/eval_10.json` |
| `robocasa_stand_mixer_peak` / `PickPlaceCounterToStandMixer` | `a100_smolvlm_flow_5min_seed0` | learned frozen SmolVLM flow | 10 | 0 | 0 | `runs/autorobobench/robocasa_stand_mixer_peak/a100_smolvlm_flow_5min_seed0/eval_10.json` |
| `robocasa_recap_offline` / `PickPlaceCounterToStandMixer` | `stand_mixer_a100_5min_init` | learned BC init | 10 | 0 | 0 | `runs/autorobobench/robocasa_recap_offline/stand_mixer_a100_5min_init/eval_10.json` |
| `video_policy_transfer` / BC5 tasks | `scarce_paired_bc` | learned transfer policy | 5 | 0 | 0 | `runs/autorobobench/video_policy_transfer/scarce_paired_bc/eval_smoke_1per_task.json` |
| `video_policy_transfer` / BC5 tasks | `smolvlm_5min` | learned transfer policy | 5 | 0 | 0 | `runs/autorobobench/video_policy_transfer/smolvlm_5min/eval_smoke_1per_task.json` |
| `video_policy_transfer` / BC5 tasks | `vit_act_5min` | learned transfer policy | 5 | 0 | 0 | `runs/autorobobench/video_policy_transfer/vit_act_5min/eval_smoke_1per_task.json` |

## Current BC5 Per-Task Rates

The current BC5 split is `OpenDrawer`, `CloseDrawer`, `CloseFridge`,
`TurnOffStove`, `PickPlaceCounterToCabinet`.

| Policy | Overall /100 | OpenDrawer | CloseDrawer | CloseFridge | TurnOffStove | PickPlaceCounterToCabinet | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| `autoresearch_full_history_act_seed0`, fixed inference | 8 | 0 | 40 | 0 | 0 | 0 | `runs/autorobobench/robocasa_bc5/autoresearch_full_history_act_seed0/eval_10_per_task_after_horizon_fix_parallel_10w.json` |
| `bc_simple_chunk32_progress750_commit8_240s_seed0` | 0 | 0 | 0 | 0 | 0 | 0 | `runs/autorobobench/robocasa_bc5/bc_simple_chunk32_progress750_commit8_240s_seed0/eval_10_per_task_parallel_10w.json` |

The best historical BC5 aggregate result in the current manifest is
`autoresearch_clip_recede4_open_only` at 14/100 (`7/50`). That eval JSON is
stored at
`runs/autorobobench/robocasa_bc5/autoresearch_clip_recede4_open_only/eval_10_per_task_local.json`.
It predates the latest exact current-split reporting and should be treated as a
historical aggregate, not the clean per-task table above.

## Auxiliary Metrics

`robocasa_progress_predictor` is not a rollout-success benchmark. The current
state/action progress predictor run completed in 7.7 seconds with:

- validation MAE: `0.111955`
- validation RMSE: `0.192321`
- validation R2: `0.557839`
- task-mean baseline RMSE: `0.289224`

Source:
`data/autorobobench/robocasa_progress_predictor_bc5_state_action_5min_metrics.json`
