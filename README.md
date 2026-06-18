# nano-robot-worlds

Offline-first robotics autoresearch. The current implementation is LIBERO-first,
with a new RoboCasa-5 learned-evaluator track for testing whether fast learned
world-model scores can replace most slow simulator rollouts during policy search.

## AutoroboBench v0

This repo now includes an AutoroboBench v0 scaffold:

```text
configs/autorobobench_v0.json
docs/autorobobench_v0.md
program_autorobobench.md
autorobobench/
tasks/robocasa_bc5/
```

AutoroboBench scores robotics research loops, not just final robot skill. The
v0 suite defines six tracks: RoboCasa BC-5, Long-Horizon RoboCasa, World Model
Evaluator, External Data, Sim RL, and Transfer/Robustness/Language. Phase 1 is
the launchable profile for the current repo: RoboCasa BC-5, Long-Horizon
RoboCasa, and World Model Evaluator.

Inspect and smoke-test the benchmark contract:

```bash
python -m autorobobench.cli describe --config configs/autorobobench_v0.json
python -m autorobobench.cli score \
  --config configs/autorobobench_v0.json \
  --results examples/autorobobench_v0_results.json
python -m autorobobench.cli hash-manifest \
  --config configs/autorobobench_v0.json \
  --out runs/autorobobench/v0_immutable_hashes.json
```

Run the executable RoboCasa BC-5 track:

```bash
python tasks/robocasa_bc5/setup.py --verify

python tasks/robocasa_bc5/train.py \
  --out-dir runs/autorobobench/robocasa_bc5/baseline \
  --train-episodes-per-task 4 \
  --val-episodes-per-task 2 \
  --steps 200

python tasks/robocasa_bc5/eval.py \
  --policy runs/autorobobench/robocasa_bc5/baseline/policy_best.pt \
  --out runs/autorobobench/robocasa_bc5/baseline/eval_success.json \
  --eval-episodes-per-task 1 \
  --render-dir runs/autorobobench/robocasa_bc5/baseline/rollouts \
  --render-episodes-per-task 1
```

The frozen public split is `data/autorobobench/robocasa_bc5_splits.json`:
episodes `0-79` train, `80-89` validation, and `90-99` eval for each of the
five RoboCasa tasks.

The task package follows the benchmark contract:

- `tasks/robocasa_bc5/setup.py`: one-time data/environment verifier
- `tasks/robocasa_bc5/train.py`: editable training entrypoint
- `tasks/robocasa_bc5/inference.py`: policy interface loaded by eval
- `tasks/robocasa_bc5/eval.py`: immutable evaluator that loads one model and
  calls `inference.act(...)`

The canonical public BC-5 starter is a chunked BC h16/w512 policy trained on
80 demos per task for 3000 steps. It scores `5/50 = 10%` on the frozen public
eval: `1/10` OpenDrawer, `4/10` CloseDrawer, and `0/10` on the three harder
manipulation tasks.

The active v0 question is:

```text
Can an AI researcher improve a tiny robot learning stack using cheap offline
losses, while final selection is grounded by closed-loop robot task success?
```

The repo is now LIBERO-only. Old toy/MuJoCo/ALOHA/ARX/mobile-manipulation
scaffolding has been removed.

## Current Stack

- LIBERO-Object seed benchmark with 5 paired-action tasks.
- Extra LIBERO video-only demonstrations for tokenizer/world-model training.
- 50 paired-action demos and 5,000 effective video-only demos by default.
- Tiny image tokenizer.
- NanoGPT-style video-token world model.
- Inverse dynamics model.
- Tiny transformer BC policy.
- Conditional flow-matching action policy.
- Fixed closed-loop LIBERO success evaluation.
- JSONL autoresearch ledger for commit/change/metric graphing.

## RoboCasa Learned-Evaluator Track

The next launchable claim is:

```text
Can a fast learned world-model evaluator rank robot policy variants well enough
to reduce slow RoboCasa simulator rollouts?
```

The RoboCasa track lives in:

```text
research/robocasa_world_evaluator.md
program_robocasa.md
configs/robocasa5_world_evaluator.json
```

Create the seed manifest:

```bash
python data/make_robocasa5.py --out data/robocasa5/manifest.json
```

Download the official RoboCasa datasets for those tasks:

```bash
python data/download_robocasa.py --manifest data/robocasa5/manifest.json --split pretrain --source human
python data/make_robocasa5.py --out data/robocasa5/manifest.json --verify-exists
```

Analyze a learned-evaluator calibration archive:

```bash
python eval/eval_world_model_ranking.py \
  --candidates runs/robocasa5/candidates.jsonl \
  --out runs/robocasa5/evaluator_ranking_metrics.json \
  --plot runs/robocasa5/evaluator_correlation.svg
```

Candidate rows should include `learned_score`, `sim_success`,
`learned_eval_rollouts`, `learned_eval_seconds`, `sim_eval_rollouts`, and
`sim_eval_seconds`. The script reports Spearman rank correlation, Pearson
correlation, top-k hit rate, and measured evaluator-vs-sim speedup.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[libero,plot]"
```

Fetch LIBERO and download LIBERO-Object:

```bash
python data/download_libero.py --dataset libero_object --use-huggingface
```

Create the seed benchmark:

```bash
python data/make_libero5.py --libero-root third_party/LIBERO --paired-demos-per-task 10
python data/split_video_and_paired.py --manifest data/libero_object5/manifest.json
```

The local LIBERO-Object raw pool contains 500 video demos. The default prep
uses `video_repeat_factor = 10`, so tokenizer/world-model training sees 5,000
effective video-only demos while paired-action data stays at 50 demos. The
video shard stores raw frames once plus a virtual sample index, so this does not
inflate the shard by 10x on disk.

This produces:

```text
data/libero_object5/manifest.json
data/libero_object5/libero_object5_video.npz
data/libero_object5/libero_object5_paired.npz
```

## Training

The main editable surface for autoresearch is `train.py`.

Run the current accepted candidate:

```bash
python train.py --out-dir runs/libero/train_py --max-train-seconds 300
```

Run fixed config experiments:

```bash
python research/run_experiment.py --config configs/libero_v0_bc.json --skip-success-eval
python research/run_experiment.py --config configs/libero_v0_world_inverse.json --skip-success-eval
```

Run the 50-iteration strategy sweep:

```bash
python research/autoresearch50.py \
  --iterations 50 \
  --max-train-seconds 300 \
  --baseline runs/libero/real_bc_all5
```

## Evaluation

Offline metric table:

```bash
python eval/eval_offline.py --runs-root runs/libero
```

Closed-loop LIBERO success:

```bash
python eval/eval_libero_success.py \
  --policy runs/libero/flow_matching_wrist_dropout/policy.pt \
  --episodes-per-task 1 \
  --max-steps 150
```

Render a closed-loop rollout:

```bash
python eval/render_libero_rollout.py \
  --policy runs/libero/flow_matching_wrist_dropout/policy.pt \
  --task-id 0 \
  --episode-id 0 \
  --max-steps 150 \
  --out runs/libero/flow_matching_wrist_dropout/eval_task0_ep0.mp4
```

## Autoresearch Artifacts

The main sweep ledger is:

```text
runs/libero/autoresearch50/ledger.jsonl
```

It records:

- iteration
- change description
- metrics
- accepted/rejected decision
- current best metrics
- run directory

The current accepted best is the flow-matching policy:

```text
runs/libero/flow_matching_wrist_dropout/policy.pt
```

It improved validation BC loss over the previous BC wrist-dropout policy, while
closed-loop success is still the primary unresolved target.

## Autoresearch Contract

For agent runs, `program.md` is the human-editable instruction file. The core
rule is:

- edit `train.py` for candidate strategies
- do not edit fixed data prep, judge, or eval code inside an experiment
- always train under the configured wall-clock budget
- evaluate after training
- commit only accepted improvements
