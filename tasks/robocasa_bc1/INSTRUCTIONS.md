# robocasa_bc1 Instructions

Keep scored runs under 300 seconds. Write outputs under
`runs/autorobobench/robocasa_bc1/<run>/`. Do not edit eval files or
split files for scored runs.

## Iterative Research Protocol

- Do not queue up a batch of experiments to run unattended.
- Think of experiments one by one: state one hypothesis, make the smallest
  relevant change, train or run the check, inspect loss/eval output, then choose
  the next experiment from that result.
- Build experiments cumulatively on the current per-run branch. When a source
  change improves eval score or a task-relevant validation/loss signal, record
  the evidence and commit it before starting the next experiment.
- The next successful experiment should start on top of the previous successful
  committed change. If a change fails or is worse, discard it or explicitly
  supersede it before moving on.
- Never merge an agent run branch to `main`.

## Task

- Optimize one policy for `TurnOnFaucet` (`TurnOnSinkFaucet` in the RoboCasa
  manifest).
- Metric: success plus a small speed bonus on successful episodes only:
  `score = (S + 0.25 * V) / (N + 0.25)`, where `S` is the number of successful
  eval episodes, `N` is the number of eval episodes, and `V` is the mean
  `1 - steps_i / max_steps` over successful episodes. Failed episodes never
  receive speed credit.
- Data: task-specific trajectories are allowed. Generic video-only pool is
  allowed for training only.
- Test-time replay is banned. `inference.py` and the submitted checkpoint may
  not read or carry demonstration trajectories, trajectory banks, manifest/split
  files, datasets, video pools, or per-episode action arrays during eval. The
  only eval-time inputs are the checkpoint's learned weights/statistics, `obs`,
  and `task`.
- Current learned base: `robocasa_faucet_direct_bc_all_data_5min_seed0`,
  6/10 success, reported as 60/100 normalized.

## Things To Try

- Scale model width, depth, chunk horizon, and image encoder capacity up or
  down.
- Try transformer/ACT policies, CNN or ResNet backbones, residual MLP heads,
  different activations, optimizers, or learning-rate schedules.
- Try diffusion or flow matching for action chunks or next latent prediction.
- Tune commit horizon and reliability/speed tradeoffs. Speed credit only comes
  from successful episodes, so all success improvements help.
- Tune training throughput and stability: batch size, gradient accumulation,
  mixed precision (`bf16`, `fp16`, or `fp32`), `torch.compile`, dataloader
  workers, caching/precompute, and checkpoint/eval cadence.

## Train

```bash
python3 tasks/robocasa_bc5/train.py \
  --manifest data/autorobobench/robocasa_bc1_manifest.json \
  --split data/autorobobench/robocasa_bc1_splits.json \
  --out-dir runs/autorobobench/robocasa_bc1/<run> \
  --policy-kind bc \
  --chunk-horizon 32 \
  --progress-conditioning \
  --progress-scale 750 \
  --eval-commit-steps 8 \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_bc1/eval_parallel.py \
  --checkpoint runs/autorobobench/robocasa_bc1/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_bc1/<run>/eval_100.json \
  --eval-episodes-per-task 100 \
  --max-steps 750 \
  --commit-steps 8 \
  --workers 28 \
  --device cuda
```

## Visualize

Summarize eval/training outputs under `<run>/visualize/`. Add `--render` to also save eval videos.

```bash
python3 tasks/robocasa_bc1/visualize.py \
  --run-dir runs/autorobobench/robocasa_bc1/<run>
```
