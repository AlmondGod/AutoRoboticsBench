# RoboAutoresearch Bench Instructions

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
