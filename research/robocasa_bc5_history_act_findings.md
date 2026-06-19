# RoboCasa BC-5 History ACT Findings

Date: 2026-06-19

## Change

Added a benchmark-compliant `history_act` policy:

- current RGB agent+wrist views
- previous RGB agent+wrist views
- current proprio
- previous proprio
- proprio delta
- task id
- ACT-style learned action queries for a 16-step action chunk

Inference keeps the previous observation internally and resets it when the frozen evaluator moves to a new task or episode. The immutable eval code is unchanged and the policy receives no test-time demonstrations.

## Run

Run directory:

`runs/autorobobench/robocasa_bc5_history_act/histact_h16_w256_s16_3min_seed0`

Training setup:

- 80 train demos/task
- 10 val demos/task
- chunk horizon: 16
- history stride: 16
- width: 256
- transformer depth: 2
- action depth: 2
- optimizer cap: 180 seconds

Metrics:

| metric | value |
| --- | ---: |
| optimizer steps | 1036 |
| train seconds | 184.5 |
| final val action MSE | 0.2613 |
| quick eval, commit 16 | 2/10 |
| quick eval, commit 8 | 2/10 |

Quick eval means first two frozen eval episodes for each of the five BC-5 tasks, `max_steps=260`.

Per-task quick eval:

| task | commit 16 | commit 8 |
| --- | ---: | ---: |
| OpenDrawer | 0/2 | 0/2 |
| CloseDrawer | 2/2 | 2/2 |
| PickPlaceCounterToStove | 0/2 | 0/2 |
| TurnOffStove | 0/2 | 0/2 |
| PickPlaceCounterToCabinet | 0/2 | 0/2 |

## Interpretation

This is an improvement over the previous sequence-flow policy, which scored `0/10` on the same quick eval. It also beats the old CNN baseline on this exact first-two-episodes slice, where the old baseline scored `0/10`.

It is not yet a confirmed replacement for the old CNN baseline because that baseline's full 50-episode eval scored `5/50`. A full 50-episode history-ACT eval is needed before calling this an accepted benchmark improvement.

Shorter receding-horizon execution (`commit_steps=8`) made CloseDrawer finish faster but did not increase the success count.

## Follow-Up

Likely next useful runs:

- full 50-episode eval for this checkpoint
- train history ACT with width 512 if compute allows
- add temporal ensembling inside `act`
- add task-progress or contact auxiliary losses for hard pick/place and stove tasks
