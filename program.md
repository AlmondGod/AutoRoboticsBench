# Autoresearch Program

You are a robotics research agent.

Your objective is to improve held-out evaluation score under a simulated
sim-to-real gap.

## Editable Files

You may edit:

- `research.py`

You may not edit:

- `robotbench/`
- `run_experiment.py`
- `judge.py`
- `plot_progress.py`
- `robotbench/tasks/*.yaml`
- existing run artifacts

## Research Rules

Each change should have a short hypothesis. Prefer small, testable changes over
large rewrites. Do not exploit benchmark implementation details, remove safety
penalties, change task definitions, or specialize to evaluation seeds.

The training simulator is not the evaluation simulator. Your goal is robust
policy learning, not overfitting the training world.

## Acceptance

A change is considered useful only if `judge.py` accepts it against the current
baseline. Safety regressions can reject a change even when task success improves.

