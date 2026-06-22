"""Task registry for the minimal harness."""

from __future__ import annotations


TASK_REGISTRY = {
    "robocasa_bc5": {
        "name": "RoboCasa BC5",
        "num_episodes": 10,
        "metric": "success_rate",
    }
}
