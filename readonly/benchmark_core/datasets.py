"""Placeholder benchmark datasets."""

from __future__ import annotations


def load_demos(task_name: str, split: str = "train") -> list[dict]:
    return [
        {"task_name": task_name, "split": split, "episode": 0, "success": True},
        {"task_name": task_name, "split": split, "episode": 1, "success": False},
    ]
