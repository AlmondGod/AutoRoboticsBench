# robocasa_language_following Instructions

Write outputs under
`runs/autorobobench/robocasa_language_following/<run>/`. Do not edit
eval files or split files for scored runs.

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

- Optimize one language-conditioned policy over four variants:
  `ChooseMeasuringCupLeftLarger`, `ChooseMeasuringCupLeftSmaller`,
  `ChooseMeasuringCupRightLarger`, `ChooseMeasuringCupRightSmaller`.
- Metric: rollout success rate under the correct language prompt. Also report
  wrong-language success and conditioning gap as diagnostics.
- Default eval: 100 total correct-language rollouts, 25 per language variant.
- Do not collapse variants into one unlabeled task.
- Test-time inference may not read manifests, splits, datasets, video pools, or
  replay stored trajectories. `inference.py` may use only checkpoint
  weights/statistics plus the current `obs`, `task`, and language prompt.

## Things To Try

- Use stronger language-conditioned fusion such as FiLM, cross-attention, or
  explicit task embeddings.
- Add contrastive language-vision alignment.
- Add hard negatives or wrong-language penalties to improve the conditioning
  gap.
- Balance variants and augment language paraphrases while preserving labels.
- Separate a shared visual encoder from language-conditioned action heads.
- Tune training throughput and stability: batch size, gradient accumulation,
  mixed precision (`bf16`, `fp16`, or `fp32`), `torch.compile`, dataloader
  workers, caching/precompute, and checkpoint/eval cadence.

## Train

```bash
python3 tasks/robocasa_language_following/train.py \
  --manifest data/autorobobench/robocasa_language_following_manifest.json \
  --split data/autorobobench/robocasa_language_following_splits.json \
  --out-dir runs/autorobobench/robocasa_language_following/<run> \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_language_following/eval.py \
  --checkpoint runs/autorobobench/robocasa_language_following/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_language_following/<run>/eval.json \
  --eval-episodes-per-task 25 \
  --device cuda
```

## Visualize

Summarize eval/training outputs under `<run>/visualize/`. Add `--render` to also save eval videos.

```bash
python3 tasks/robocasa_language_following/visualize.py \
  --run-dir runs/autorobobench/robocasa_language_following/<run>
```
