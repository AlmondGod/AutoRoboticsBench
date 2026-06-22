# AutoRoboBench v0

`benchmark.json` defines the current suites and task specs. `setup.py` is the
only benchmark measurement entrypoint.

## Commands

```bash
python setup.py --describe-benchmark --suite autorobobench_v0
python setup.py --score-results path/to/results.json --suite autorobobench_v0
python setup.py --hash-manifest --suite autorobobench_v0 --out runs/autorobobench/v0_hashes.json
```

The counted `autorobobench_v0` suite contains `robocasa_bc1`,
`robocasa_visual_world_model`, and `robocasa_world_model_posttraining`.
Other task packages are available through `autorobobench_extra_v0` for
optional runs and ablations.

## Task Contract

Each task package owns:

- `setup.py`: verify generated metadata and local datasets
- `train.py`: editable training entrypoint
- `inference.py`: policy/world-model loading interface used by eval
- `eval.py`: evaluator wrapper
- `visualize.py`: editable artifact viewer that writes summaries/media under
  `<run-dir>/visualize/`
- `task.json`: task metadata, scoring contract, and immutable file list
- `INSTRUCTIONS.md`: short task-specific instructions

Generated metadata under `data/` is recreated by `python setup.py`; it is not
the source of truth. `runs/` is local output and is also not committed.

BC policy tasks use `visualize.py` to summarize eval results and can optionally
trigger render evals. World-model tasks use it to compare predicted metrics
against actual held-out metrics. Offline-RL posttraining uses it to inspect
source counts, assigned advantages, sample weights, and eval success.
