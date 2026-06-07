# nano-robot-worlds

Offline-first robotics autoresearch on LIBERO-Object-5.

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
- Tiny image tokenizer.
- NanoGPT-style video-token world model.
- Inverse dynamics model.
- Tiny transformer BC policy.
- Conditional flow-matching action policy.
- Fixed closed-loop LIBERO success evaluation.
- JSONL autoresearch ledger for commit/change/metric graphing.

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
