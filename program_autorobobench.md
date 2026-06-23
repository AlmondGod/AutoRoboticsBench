# AutoRoboBench Agent Instructions

You are running an AutoRoboBench RoboCasa research loop.

Goal:

```text
Improve the robot-learning system under the fixed benchmark budget.
```

Primary score comes from evaluator reruns, not self-reported metrics.

## Rules

- Read the active task's `task.json` and `INSTRUCTIONS.md` first.
- Do not edit files matched by the active track's `immutable_globs`.
- Do not read hidden eval files, canary files, or answer files.
- Do not use network access unless the active track explicitly allows fixed
  external-data corpora.
- Keep experiment outputs under `runs/`.
- Commit only accepted improvements.
- You may run as many experiments and evaluator reruns as you want, in any
  order, within the outer run deadline.
- Every model-training command, including custom training loops and helper
  pretraining jobs, must be capped at 300 seconds or less. Do not raise or
  bypass task `--max-train-seconds` limits.
- Run an evaluator at least once per wall-clock hour during active work and
  record each checkpoint in `runs/<RUN_ID>/timing.jsonl`. Use
  `python scripts/record_eval_checkpoint.py --run-id <RUN_ID> --eval-json <eval.json>`
  after each interim eval.
- Scored/interim policy eval checkpoints should use 100 total rollouts. For
  single-task policy tracks this is `--eval-episodes-per-task 100`; for BC5
  five-task tracks this is `--eval-episodes-per-task 20`; for language
  following this is `--eval-episodes-per-task 25`.

## Benchmark Metadata

Use `benchmark.json` for suite membership and `python setup.py` for
measurement:

```bash
python setup.py --describe-benchmark --suite autorobobench_v0
python setup.py --score-results path/to/results.json --suite autorobobench_v0
```

Generated manifests, splits, video pools, policy registries, and eval metadata
are written under `data/` by `python setup.py`.
