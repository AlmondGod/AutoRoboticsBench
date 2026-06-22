"""Task registry for the minimal harness."""

from __future__ import annotations


TASK_REGISTRY = {
    "toy_pickplace": {
        "name": "Toy Pickplace",
        "num_episodes": 10,
        "metric": "success_rate",
    }
}
