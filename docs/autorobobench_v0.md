# AutoroboBench v0

AutoroboBench is a benchmark of robotics research loops.

The question is:

```text
Given a fixed robot-learning repo, data budget, compute budget, and hidden eval,
can an agent make the robot system better?
```

It is not primarily a benchmark of final robot skill. It scores autonomous
research progress: reading the repo, choosing experiments, editing code, running
bounded trials, keeping improvements, and surviving a clean hidden rerun.

## Tracks

`configs/autorobobench_v0.json` defines six tracks and a 100 point score.

| ID | Track | Phase | Points | Primary metric |
| --- | --- | ---: | ---: | --- |
| `robocasa_bc5` | RoboCasa BC-5 | 1 | 15 | hidden held-out success |
| `robocasa_long_horizon` | Long-Horizon RoboCasa | 1 | 20 | full success plus subgoal progress |
| `world_model_evaluator` | World Model Evaluator | 1 | 20 | policy-ranking correlation and speedup |
| `external_data_wall` | External Data / Data Wall | 2 | 15 | value of fixed external robot data |
| `sim_rl_improvement` | Sim RL Improvement | 3 | 15 | success AUC under sim budget |
| `transfer_robust_language` | Transfer / Robustness / Language | 3 | 15 | hidden robust success |

Phase 1 is the launchable v0 profile for this repo: BC-5, Long-Horizon, and
World Model Evaluator. Phases 2 and 3 are defined in the score file so the suite
can grow without changing the scoring contract.

## Scoring

Each track result is a JSON object with metrics in `[0, 1]`, plus the track's
primary metric. For primary metrics with known starter/reference targets, the
score can use:

```text
normalized_progress =
  clip((agent_metric - starter_metric) / (reference_metric - starter_metric), 0, 1)
```

The suite score is:

```text
AutoroboBench score = sum(track_weight * track_score)
```

where `track_score` is a weighted component average from the config.

Run a local smoke score:

```bash
python -m autorobobench.cli score \
  --config configs/autorobobench_v0.json \
  --results examples/autorobobench_v0_results.json
```

## Anti-Cheating Contract

Final scoring should be done by organizer rerun, not by self-reported metrics.

Protocol:

- public train/dev seeds for debugging
- hidden seeds, objects, layouts, tasks, and canary files for final eval
- immutable evaluator and data-prep files
- file hashes for hidden/eval/config artifacts
- network disabled by default, except fixed external-data tasks
- submit patch and run log, not claimed score
- statistical eval over multiple seeds
- reproducibility/integrity penalties for metric fabrication or eval tampering

Create a local immutable-file hash manifest:

```bash
python -m autorobobench.cli hash-manifest \
  --config configs/autorobobench_v0.json \
  --out runs/autorobobench/v0_immutable_hashes.json
```

## Current Repo Mapping

The current repo already has the ingredients for the Phase 1 launch profile:

- RoboCasa seed tasks: `data/robocasa5/manifest.json`
- frozen RoboCasa BC-5 split:
  `data/autorobobench/robocasa_bc5_splits.json`
- RoboCasa BC-5 training entrypoint:
  `train/train_autorobobench_robocasa_bc5.py`
- immutable RoboCasa BC-5 success eval entrypoint:
  `eval/eval_autorobobench_robocasa_bc5.py`
- BC and policy loops: `train.py`, `train/`, `models/`, `research/`
- World model evaluator traces:
  `runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl`
- Ranking metric implementation: `eval/eval_world_model_ranking.py`

The World Model Evaluator track should be scored by decision usefulness:
correlation with true sim success, ranking accuracy, calibration, and measured
speedup. Pixel/video prediction loss is diagnostic only.

## Executable RoboCasa BC-5 Track

The v0 executable BC-5 track uses the committed split file. Agents may train on
any prefix or subset of the train episode IDs, but success is measured only by
the eval IDs in the split file.

Train a small baseline:

```bash
python train/train_autorobobench_robocasa_bc5.py \
  --out-dir runs/autorobobench/robocasa_bc5/baseline \
  --train-episodes-per-task 4 \
  --val-episodes-per-task 2 \
  --steps 200
```

Evaluate and render one rollout per task:

```bash
python eval/eval_autorobobench_robocasa_bc5.py \
  --policy runs/autorobobench/robocasa_bc5/baseline/policy_best.pt \
  --out runs/autorobobench/robocasa_bc5/baseline/eval_success.json \
  --eval-episodes-per-task 1 \
  --render-dir runs/autorobobench/robocasa_bc5/baseline/rollouts \
  --render-episodes-per-task 1
```

Visualize an agent run ledger:

```bash
python -m autorobobench.plot_robocasa_bc5 \
  --ledger runs/autorobobench/robocasa_bc5_codex/experiments.jsonl \
  --out-dir runs/autorobobench/robocasa_bc5_codex/plots
```

### Canonical Starter

The current public BC-5 starter is the chunked BC policy trained with:

```bash
python train/train_autorobobench_robocasa_bc5.py \
  --out-dir runs/autorobobench/robocasa_bc5_starter/bc5_h16_w512_80demo_seed0 \
  --train-episodes-per-task 80 \
  --val-episodes-per-task 10 \
  --chunk-horizon 16 \
  --frame-stride 1 \
  --steps 3000 \
  --batch-size 128 \
  --width 512 \
  --dropout 0.02 \
  --lr 2e-4 \
  --image-noise 0.005 \
  --proprio-noise 0.005 \
  --action-smooth 0.0005 \
  --chunk-decay 0.8 \
  --seed 0
```

Measured public eval result:

```text
overall: 5/50 = 10%
OpenDrawer: 1/10
CloseDrawer: 4/10
PickPlaceCounterToStove: 0/10
TurnOffStove: 0/10
PickPlaceCounterToCabinet: 0/10
```

This is intentionally nonzero but weak. The benchmark should measure improving
from `10%` toward stronger cross-task manipulation, not escaping a broken zero
baseline.
