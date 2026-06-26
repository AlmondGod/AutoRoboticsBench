# Codex H100 Benchmark Run - 2026-06-26

Minimal committed summary for pooling benchmark results across agents.

Run setup:
- Agent: Codex
- Model label: `gpt-5.5`
- Scaffold: `h100-codex-cli`
- Hardware: H100 RunPod
- Budget: 4 tasks, 2 hours per task

Files:
- `task_summary.csv`: one row per final valid task run.
- `summed_score_by_time.csv`: summed 4-task benchmark score at 30, 60, 90, and 120 minutes per task.
- `summed_score_by_time.png`: plot generated from `summed_score_by_time.csv`.

Score convention:
- At each time cutoff, use the best score reached by each task up to that per-task elapsed time.
- Sum the four task scores to get `sum_score`.
- This avoids using cumulative task order as the x-axis and makes future agent runs comparable.
