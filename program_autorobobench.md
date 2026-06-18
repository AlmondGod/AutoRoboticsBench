# AutoroboBench Agent Instructions

You are running an AutoroboBench v0 research loop.

Goal:

```text
Improve the robot-learning system under the fixed benchmark budget.
```

Primary score comes from hidden evaluator reruns, not self-reported metrics.

## Rules

- Do not edit files matched by the active track's `immutable_globs` in
  `configs/autorobobench_v0.json`.
- Do not read hidden eval files, canary files, or answer files.
- Do not use network access unless the active track explicitly allows fixed
  external-data corpora.
- Keep a clear experiment ledger with change, commit, train budget, metrics,
  accepted/rejected decision, and notes.
- Prefer changes that can survive a clean rerun.
- Commit only accepted improvements.

## Phase 1 Tracks

### RoboCasa BC-5

Improve the BC/VLM policy on the five seed RoboCasa tasks.

Executable v0 files:

- frozen split: `data/autorobobench/robocasa_bc5_splits.json`
- setup verifier: `tasks/robocasa_bc5/setup.py`
- editable train entrypoint: `tasks/robocasa_bc5/train.py`
- editable inference interface: `tasks/robocasa_bc5/inference.py`
- immutable eval entrypoint: `tasks/robocasa_bc5/eval.py`

Quick dev command:

```bash
python tasks/robocasa_bc5/setup.py --verify

python tasks/robocasa_bc5/train.py \
  --out-dir runs/autorobobench/robocasa_bc5_dev/exp001 \
  --train-episodes-per-task 1 \
  --val-episodes-per-task 1 \
  --steps 50 \
  --frame-stride 16 \
  --width 64 \
  --chunk-horizon 4 \
  --device cpu

python tasks/robocasa_bc5/eval.py \
  --policy runs/autorobobench/robocasa_bc5_dev/exp001/policy_best.pt \
  --out runs/autorobobench/robocasa_bc5_dev/exp001/eval_success.json \
  --eval-episodes-per-task 1 \
  --max-steps 40 \
  --commit-steps 4 \
  --device cpu
```

For a real run, increase train episodes, steps, model width, and max eval steps.
Do not change the eval split or immutable eval script.

Good experiment families:

- action chunking and temporal ensembling
- flow or diffusion action heads
- better language/task conditioning
- image augmentation and view dropout
- balanced multitask sampling
- auxiliary progress/value losses
- validation-gated early stopping

### Long-Horizon RoboCasa

Improve compositional manipulation and recovery.

Good experiment families:

- subgoal prediction
- progress-value heads
- open-loop chunk plus closed-loop correction
- failure recovery policy
- task decomposition from language

### World Model Evaluator

Train a learned evaluator that ranks policy candidates faster than simulator
rollouts.

Good experiment families:

- trace-conditioned latent dynamics
- progress/success calibration
- held-out candidate splits
- speed/accuracy tradeoffs
- ranking loss instead of only pixel or latent loss

The World Model Evaluator track is judged by policy-ranking usefulness, not
visual fidelity alone.

## Required Ledger Fields

Each experiment row should contain:

```json
{
  "experiment": 1,
  "commit": "abc1234",
  "track": "robocasa_bc5",
  "change": "increase temporal chunk horizon from 4 to 8",
  "train_budget_seconds": 300,
  "run_dir": "runs/autorobobench/robocasa_bc5_dev/exp001",
  "history": "runs/autorobobench/robocasa_bc5_dev/exp001/history.json",
  "eval": "runs/autorobobench/robocasa_bc5_dev/exp001/eval_success.json",
  "metrics": {
    "success_rate": 0.2,
    "val_loss": 0.9
  },
  "accepted": true,
  "notes": "Improves frozen dev success without touching eval."
}
```

## Scoring Smoke Test

```bash
python -m autorobobench.cli describe --config configs/autorobobench_v0.json
python -m autorobobench.cli score \
  --config configs/autorobobench_v0.json \
  --results examples/autorobobench_v0_results.json
```
