"""Placeholder benchmark environments."""

from __future__ import annotations


def make_env(task_name: str, split: str, seed: int = 0) -> dict:
    return {
        "task_name": task_name,
        "split": split,
        "seed": seed,
        "kind": "toy_env",
    }
