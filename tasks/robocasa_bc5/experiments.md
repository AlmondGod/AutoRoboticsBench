# RoboCasa BC5 Experiments

## SmolVLM2 frozen vision backbone

Date: 2026-06-20

Question: does `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` improve BC5 when used as a frozen VLM backbone for the existing chunked BC/flow policy?

Setup:
- Data: 5 RoboCasa BC5 tasks, 4 train demos/task, 2 val demos/task.
- Training: 5 minute action-head budget, `chunk_horizon=16`, `frame_stride=2`, `width=256`, `action_depth=3`.
- Eval: normalized validation action MSE plus 1 closed-loop episode/task locally.

| Backbone | Frozen params | Trainable params | Cache train/val | Best val MSE | Final val MSE | Closed-loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolVLM2-500M vision tower | 507.5M | 7.7M | 22.3s / 10.3s | 0.4785 | 0.4941 | 1/5 |
| CLIP ViT-B/32 | 151.3M | 7.0M | 7.1s / 2.4s | 0.4921 | 0.4990 | 1/5 |

Result: SmolVLM2 gave a small offline MSE improvement over CLIP, but did not improve the tiny closed-loop sample; both solved only `CloseDrawer`.

Follow-up closed-loop eval with 10 episodes/task:

| Backbone | OpenDrawer | CloseDrawer | PickPlaceCounterToStove | TurnOffStove | PickPlaceCounterToCabinet | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolVLM2-500M vision tower | 0/10 | 6/10 | 0/10 | 0/10 | 0/10 | 6/50 |
| CLIP ViT-B/32 | 0/10 | 6/10 | 0/10 | 0/10 | 0/10 | 6/50 |

Conclusion: the larger SmolVLM2 frozen vision tower improves validation MSE slightly, but this did not transfer to measured closed-loop success under the 5-minute training budget. The current benchmark signal is dominated by `CloseDrawer`; the other four tasks remain unsolved by both policies.

## Autoresearch pass: execution policy and full-data baselines

Date: 2026-06-20

Baseline to beat: `CLIP ViT-B/32`, 5 minute training, 4 demos/task, 6/50 closed-loop success.

| # | Change | Offline signal | OpenDrawer | CloseDrawer | PickPlaceCounterToStove | TurnOffStove | PickPlaceCounterToCabinet | Total | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Full 80 demos/task, history ACT, progress conditioning, per-task action normalization, 5 minute train | val MSE 0.3013 | 0/10 | 4/10 | 0/10 | 0/10 | 0/10 | 4/50 | discard |
| 2 | Mean action ensemble of prior CLIP and SmolVLM2 frozen policies | n/a | 0/10 | 6/10 | 0/10 | 0/10 | 0/10 | 6/50 | discard, tied |
| 3 | CLIP receding horizon on all tasks, return 4 actions per 16-action chunk | partial eval | 1/10 | 4/9 when stopped | - | - | - | - | discard, hurt CloseDrawer |
| 4 | CLIP task-specific receding horizon: OpenDrawer returns 4 actions, other tasks return 16 | n/a | 1/10 | 6/10 | 0/10 | 0/10 | 0/10 | 7/50 | keep |

Kept finding: a small ACT-style receding-horizon execution change improved the actual closed-loop benchmark without retraining. Applying faster replanning only to `OpenDrawer` found one new success while preserving the previous `CloseDrawer` success rate. Applying the same receding horizon to every task hurt `CloseDrawer`, so execution cadence should be task-aware until the policy is stronger.

Kept checkpoint: `runs/autorobobench/robocasa_bc5/autoresearch_clip_recede4_open_only/policy_best.pt`
Kept eval JSON: `runs/autorobobench/robocasa_bc5/autoresearch_clip_recede4_open_only/eval_10_per_task_local.json`
