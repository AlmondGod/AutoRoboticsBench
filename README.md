# robot-autoresearch

Phase 1 robotics autoresearch harness.

This repository is a small, constrained robotics research loop inspired by
Karpathy's `autoresearch`, adapted for simulated sim-to-real gaps. The agent's
editable surface is intentionally narrow: future autoresearch agents should edit
`research.py` while the benchmark, evaluator, judge, and task definitions remain
fixed.

## Phase 1 Scope

- Lightweight continuous-control robot benchmark.
- Separate training and evaluation worlds.
- Evaluation world has shifted dynamics, latency, noise, and object parameters.
- Fixed scoring with safety penalties.
- JSON run artifacts plus a repository-level `research_log.jsonl`.
- Plotting utility for commit/change/score progress over time.

Phase 1 intentionally excludes real robots, internet data, behavior cloning
demos, vision policies, and external datasets.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

python run_experiment.py --task reach --budget-seconds 5 --seeds 0 1
python run_experiment.py --task push --budget-seconds 5 --seeds 0 1 --change-note "baseline push smoke"
python plot_progress.py
```

Run artifacts are written to `runs/<run-id>/`. The long-lived progress ledger is
`runs/research_log.jsonl`.

## MuJoCo Backend

The default backend is the lightweight toy simulator. A MuJoCo backend is also
available for real rigid-body physics, MJCF models, contact, and camera
rendering.

Install the optional dependency:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[mujoco]"
```

Run MuJoCo evaluation:

```bash
python run_experiment.py --backend mujoco --task reach --budget-seconds 60 --seeds 0 1 2
python run_experiment.py --backend mujoco --task push --budget-seconds 60 --seeds 0 1 2
```

Render a MuJoCo camera video:

```bash
python render_eval_video.py --backend mujoco --task reach --out runs/mujoco_eval_reach.mp4
```

The MJCF model lives at
`robotbench/assets/mujoco/planar_arm.xml`.

## ALOHA 14-Actuator Backend

The `aloha` backend uses the real ALOHA bimanual robot model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). The
model has 16 physical joints and a 14-actuator control interface: 6 arm joints
plus one gripper actuator per side.

Fetch the Apache-2.0 Menagerie assets:

```bash
python scripts/fetch_menagerie.py --model aloha
```

Run and render ALOHA:

```bash
python run_experiment.py --backend aloha --task reach --budget-seconds 60 --seeds 0 1 2
python render_eval_video.py --backend aloha --task reach --out runs/aloha_eval_reach.mp4
```

The fetched assets are stored under `third_party/mujoco_menagerie/` and are
ignored by git. Keep using the fetch script so the source and license are clear.

## Mobile ALOHA Status

Actual Mobile ALOHA is not implemented yet. The public
[Mobile ALOHA repository](https://github.com/MarkFzp/mobile-aloha) is a
ROS/hardware/data-collection codebase, not a clean MuJoCo asset package that can
be dropped into this benchmark.

`mobile_aloha_mock` is only a placeholder mobile-manipulation setting built from
the Menagerie ALOHA model. It mounts the ALOHA arms on a kinematic mobile base
and adds base translation commands before the 14 ALOHA controls:

```text
action[0:3]   mobile base command
action[3:17]  ALOHA 14-actuator command
```

This is not Mobile ALOHA and is not an official Menagerie robot asset. It is a
temporary benchmark environment for testing mobile manipulation research loops
until we import or author a real Mobile ALOHA MJCF.

```bash
python scripts/fetch_menagerie.py --model aloha
python run_experiment.py --backend mobile_aloha_mock --task reach --budget-seconds 60 --seeds 0 1 2
python render_eval_video.py --backend mobile_aloha_mock --task reach --out runs/mobile_aloha_mock_eval_reach.mp4
```

## Main Commands

Run one candidate:

```bash
python run_experiment.py --backend toy --task reach --budget-seconds 60 --seeds 0 1 2
```

Compare a candidate to a baseline:

```bash
python judge.py --baseline runs/<baseline-id> --candidate runs/<candidate-id>
```

Create a progress plot. SVG output works with only the Python standard library;
PNG output uses Matplotlib when it is installed:

```bash
python plot_progress.py --log runs/research_log.jsonl --out runs/progress.svg
```

## Autoresearch Contract

The benchmark assumes future agents may edit only `research.py`. See
`program.md` for the full operating contract.
