# robot-autoresearch

Compact AutoRoboBench harness for RoboCasa robot-learning research loops.

The source tree is intentionally small:

- `benchmark.json`: benchmark suites, tracks, weights, metrics, and task specs
- `setup.py`: universal installer/verifier, generated metadata writer, scorer, and hasher
- `tasks/`: task-owned setup, train, inference, eval, visualize, model, and instruction files
- `data/`: local generated benchmark metadata plus shipped pretrained policy artifacts
- `benchmark_harness/`: read-only eval harness mounted for agents as `/workspace/read_only`

`configs/`, `examples/`, repo-level `models/`, repo-level `train/`, and the
`autorobobench` Python package were removed. Task implementations own their
training/model code directly.

## Install

From a fresh checkout:

```bash
git clone <repo-url> robot-autoresearch
cd robot-autoresearch
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[robocasa,plot]"
python setup.py
```

`python setup.py` writes generated JSON metadata under `data/`, validates the
benchmark spec, checks imports, and runs metadata-only task setup. It does not
require the full RoboCasa datasets.

To download referenced RoboCasa tasks:

```bash
python setup.py --download-robocasa --yes
```

To verify mounted or synced datasets:

```bash
python setup.py --verify
```

## Benchmark Commands

Inspect the main suite:

```bash
python setup.py --describe-benchmark --suite autorobobench_v0
```

Score a result file:

```bash
python setup.py --score-results path/to/results.json --suite autorobobench_v0
```

Hash immutable benchmark files:

```bash
python setup.py --hash-manifest --suite autorobobench_v0 --out runs/autorobobench/v0_hashes.json
```

Additional task packages are grouped under `autorobobench_extra_v0` for
optional, non-main-score runs.

## Docker Harness

The Docker harness creates a fresh training container for an external agent,
collects artifacts, runs a simple rule-based judge, and evaluates the final
submission in a separate clean eval container. The examples below use
`robocasa_bc5` as a lightweight real RoboCasa task for smoke testing the
harness.

Build the image:

```bash
./docker/build.sh
```

Start a run:

```bash
python scripts/launch_agent_run.py --agent codex --task robocasa_bc5 --base dummy --seed 0
```

Copy the printed prompt into Codex, Claude, Gemini, or another external agent.
The run launcher also writes the same prompt to `runs/<RUN_ID>/prompt.txt`.

Run commands through the generated wrapper:

```bash
runs/<RUN_ID>/run.sh "cd /workspace/task && python train.py"
```

Collect artifact metadata and snapshots:

```bash
python scripts/collect_artifacts.py --run-id <RUN_ID>
```

Run the rule-based judge:

```bash
python scripts/judge_run.py --run-id <RUN_ID> --task robocasa_bc5
```

Evaluate in a clean eval container:

```bash
RUN_ID=<RUN_ID> TASK=robocasa_bc5 ./docker/run_eval_container.sh
```

The Docker eval path mounts the repo read-only and runs the same real task
evaluator used by RunPod finalization. It expects
`runs/<RUN_ID>/output/final_submission/` to contain a checkpoint such as
`policy_best.pt`, plus `inference.py` if the submission overrides the default
task inference.

## RunPod Dockerless Mode

RunPod Pods are already containers, so they usually cannot launch nested Docker
training/eval containers. For RunPod development runs, use the dockerless
launcher. This keeps the same `/workspace/task`, `/workspace/output`, and
`/workspace/logs` paths by symlinking them to `runs/<RUN_ID>/`.

On a fresh RunPod GPU pod:

```bash
cd /workspace
git clone <repo-url> autoroboticsbench
cd /workspace/autoroboticsbench
./scripts/runpod_prepare_run.sh \
  --agent codex \
  --model gpt-5-codex \
  --scaffold manual \
  --task robocasa_bc5 \
  --base dummy \
  --seed 0
```

The prepare command bootstraps Python dependencies, checks GPU visibility, runs
the benchmark artifact preflight, creates a run directory, writes
`runs/<RUN_ID>/prompt.txt`, and prints the exact commands to use next. If setup
is already done, pass `--skip-setup`.

Each task provides a minimal sandbox template at
`tasks/<TASK>/workspace_template/`. The launcher copies exactly these files into
`runs/<RUN_ID>/task/`, which is exposed to the agent as `/workspace/task`:

- `task.md`: read-only task guidance
- `train.py`: editable training entrypoint
- `inference.py`: editable inference/submission entrypoint

Eval/scoring files, split files, datasets, and read-only benchmark helpers stay
outside the writable task sandbox. If an agent edits `train.py` or
`inference.py` and the change improves score, `scripts/commit_improvement.py`
syncs those source files back into `tasks/<TASK>/workspace_template/` before
committing. It does not sync edits to `task.md`.

RoboCasa eval dependencies can also be installed directly with:

```bash
./scripts/install_robocasa_runtime.sh
```

By default the launcher expects to start from a clean `main` branch and creates
a per-run branch named `codex/<RUN_ID>`. Agents must never merge this branch to
`main`. If a tracked source change improves eval score, commit it with:

```bash
python scripts/commit_improvement.py --run-id <RUN_ID> --task robocasa_bc5
```

The commit helper reads `runs/<RUN_ID>/eval/results.json`, refuses to commit on
`main`/`master`, stages non-`runs/` source changes, and commits only when the
score improves the run's last committed score. For local smoke tests or a dirty
checkout, pass `--no-git-branch`.

You can also run the two steps separately:

```bash
./scripts/setup_runpod_env.sh
python scripts/launch_runpod_run.py --agent codex --task robocasa_bc5 --base dummy --seed 0
```

Run commands through the generated wrapper:

```bash
runs/<RUN_ID>/run.sh "cd /workspace/task && python train.py"
```

Before finalization, place the scored submission under
`/workspace/output/final_submission`. The RunPod final evaluator requires a real
checkpoint file there, preferably `policy_best.pt`; include `inference.py` in the
same directory when your inference code differs from the task default.

The exact start message for an agent is:

```bash
Read runs/<RUN_ID>/prompt.txt and follow it exactly.
```

The wrapper records every command in `runs/<RUN_ID>/commands.jsonl` and refuses
to execute commands after the run deadline. Finalize the run with:

```bash
python scripts/finalize_run.py --run-id <RUN_ID> --task robocasa_bc5 --mode runpod
```

Finalization runs the task's real evaluator, judge, artifact collection, writes
`runs/<RUN_ID>/final_report.json`, writes aggregate-ready
`runs/<RUN_ID>/run_summary.json`, and records `runs/<RUN_ID>/finished_at.txt`.

For benchmark comparisons, agents should use the full per-task budget. With a
2-hour task budget, keep running one experiment at a time until close to the
2-hour deadline, leaving only enough time to package `final_submission`, record
the last eval checkpoint, and finalize. Do not stop early just because a first
improvement was found.

Required reference artifacts can be checked before starting a run:

```bash
python scripts/preflight_benchmark.py --suite autorobobench_v0
```

Agents should not manually edit `run_summary.json`. If token or cost usage is
available, write it before finalization to `runs/<RUN_ID>/run_usage.json`:

```json
{
  "input_tokens": 0,
  "output_tokens": 0,
  "reasoning_tokens": 0,
  "total_tokens": 0,
  "estimated_usd": 0.0
}
```

Agents must not queue up a batch of experiments to run unattended. The intended
research loop is one experiment at a time: state one hypothesis, make the
smallest relevant change, train or run the check, inspect loss/eval output, and
then choose the next experiment from that result. Experiments are cumulative on
the per-run branch. When a source change improves eval score or a task-relevant
validation/loss signal, record the evidence and commit it before starting the
next experiment so the next idea starts on top of the best branch state. If a
change fails or is worse, discard it or explicitly supersede it before moving
on.

Every model-training command, including custom training loops and helper
pretraining jobs, must remain capped at 300 seconds or less. During active
work, run an evaluator at least once per wall-clock hour and append the result
to `runs/<RUN_ID>/timing.jsonl`. Policy eval checkpoints use 100 total rollouts:
100 for single-task tracks, 20 per task for BC5 five-task tracks, and 25 per
variant for language following.

```bash
python scripts/record_eval_checkpoint.py \
  --run-id <RUN_ID> \
  --eval-json runs/<RUN_ID>/path/to/interim_eval.json \
  --label hourly
```

Each agent run branch contains only source changes; `runs/` is ignored and
stays local to the worktree or pod. For a local multi-run comparison, return to
a clean `main` checkout before preparing the next run, but keep the `runs/`
directory. For separate RunPod pods, copy or sync each completed `runs/<RUN_ID>/`
directory back to one analysis checkout before aggregating.

This mode is useful for RunPod iteration, but it is not equivalent to the clean
Docker train/eval isolation used by the benchmark harness.

## Time Budget Policy

The default v0 policy is strict per-task wall-clock time: each task run gets the
same `--timeout-hours`, resource class, and seed protocol. This is easier to
compare across agents and prevents a run from spending the whole benchmark
budget on one task.

Run the full counted RunPod suite with a 2 hour hard max per task:

```bash
python scripts/runpod_run_suite.py \
  --suite autorobobench_v0 \
  --agent codex \
  --model gpt-5-codex \
  --scaffold manual \
  --base dummy \
  --seed 0 \
  --timeout-hours 2 \
  --start-branch main \
  --skip-setup \
  --repeat-agent-command-until-timeout \
  --agent-command 'codex exec --full-auto "$(cat {prompt})"'
```

If `--agent-command` is omitted, the script prepares one branch/run per suite
task and writes a suite manifest under `runs/suites/`, but it does not run an
agent or finalize the runs. With `--agent-command`, the command is run once per
task and is killed after `--timeout-hours`; with
`--repeat-agent-command-until-timeout`, the command is re-run until the per-task
budget expires. Finalization and aggregation run after each task.

A separate portfolio track can allow one total suite budget where agents choose
how to allocate time across tasks. Report it separately because it measures both
task performance and scheduling strategy.

## Analysis

Aggregate completed runs:

```bash
python scripts/aggregate_runs.py
```

Create scaling plots:

```bash
python scripts/plot_scaling.py
```

Generated CSVs and PNGs are written under `analysis/` and ignored by git.

## Tracks

The counted `autorobobench_v0` task packages are:

| Track | Package | Main RoboCasa task/data | Evaluation metric |
| --- | --- | --- | --- |
| RoboCasa BC1 | `tasks/robocasa_bc1/` | `TurnOnFaucet` (`TurnOnSinkFaucet`) | `bc1_reliability_speed_score`: eval success plus a small speed bonus on successful episodes only |
| Visual World Model | `tasks/robocasa_visual_world_model/` | BC-5 next-frame prediction | `visual_world_model_score`: fixed-policy eval correlation plus pixel/state/progress/reward prediction |
| World-Model Posttraining | `tasks/robocasa_world_model_posttraining/` | `PickPlaceCounterToStandMixer` policy improvement | Eval rollout success rate |
| RoboCasa BC5 With Video | `tasks/robocasa_bc5_with_video/` | BC-5 demos plus RGB-only video pool | Mean eval success across BC-5 tasks |

Optional extra task packages are:

| Track | Package | Main RoboCasa task/data | Evaluation metric |
| --- | --- | --- | --- |
| RoboCasa BC-5 | `tasks/robocasa_bc5/` | `OpenCabinet`, `CloseDrawer`, `CloseFridge`, `TurnOffStove`, `PickPlaceCounterToCabinet` | Mean eval success across the five tasks |
| Long-Horizon Microwave | `tasks/robocasa_long_horizon/` | `PickPlaceCounterToMicrowave` | Eval success on the long-horizon task |
| RoboCasa Reward Model | `tasks/robocasa_world_model/` | BC-5/StandMixer transition reward and policy-ranking model | `reward_model_benchmark_score`: policy ranking/calibration plus reward/progress prediction |
| RoboCasa Language Following | `tasks/robocasa_language_following/` | measuring-cup language variants | `success_rate`: eval success under the correct language prompt |
| Offline-RL Posttraining | `tasks/robocasa_offlinerl_posttraining/` | `PickPlaceCounterToStandMixer` policy improvement | `success_rate`: eval rollout success after posttraining |

Each task owns its `setup.py`, `train.py`, `inference.py`, `eval.py`,
`visualize.py`, `task.json`, and `INSTRUCTIONS.md`. Visualizers write compact
JSON/SVG summaries, and optional render artifacts where supported, under
`runs/autorobobench/<task>/<run>/visualize/`.

## Local Outputs

`runs/` is local-only and recreated by training/eval commands. Generated JSON
metadata in `data/` is also local-only; `setup.py` recreates it from embedded
benchmark metadata. Shipped policy checkpoint artifacts live under
`data/autorobobench/pretrained_policies/`.

## Smoke Checks

Tiny BC-5 train/eval:

```bash
python tasks/robocasa_bc5/setup.py --verify

python tasks/robocasa_bc5/train.py \
  --out-dir runs/autorobobench/robocasa_bc5/baseline \
  --train-episodes-per-task 4 \
  --val-episodes-per-task 2 \
  --max-train-seconds 60

python tasks/robocasa_bc5/eval.py \
  --policy runs/autorobobench/robocasa_bc5/baseline/policy_best.pt \
  --out runs/autorobobench/robocasa_bc5/baseline/eval_success.json \
  --eval-episodes-per-task 1
```

Long-horizon wrapper:

```bash
python tasks/robocasa_long_horizon/setup.py --verify
python tasks/robocasa_long_horizon/train.py --max-train-seconds 60
```

Video-transfer wrapper:

```bash
python tasks/robocasa_bc5_with_video/setup.py --verify
python tasks/robocasa_bc5_with_video/train.py --max-train-seconds 300
```

StandMixer base policy for posttraining tasks:

```bash
python scripts/train_stand_mixer_base_until_nonzero.py --attempt-seconds 300 --device cuda
```

The shared default path for the two StandMixer posttraining tasks is
`runs/autorobobench/robocasa_stand_mixer_base/nonzero_base/policy_best.pt`.
The current promoted A100 artifact is a learned BC checkpoint that scored 2/10
on `PickPlaceCounterToStandMixer`; it was trained with an eval-included
diagnostic split and should be treated as a posttraining base artifact, not as a
fair standalone benchmark submission.

The world-model posttraining task ships a default frozen visual world model at
`data/autorobobench/pretrained_world_models/robocasa_visual_world_model_spatial_conv_11task_20min.pt`.
The `.pt` file is tracked through Git LFS. After cloning, run `git lfs pull` if
the file is still an LFS pointer. It is an 11-task VisualRoboCasaWorldModel
checkpoint trained for the default posttraining warm-start path.
