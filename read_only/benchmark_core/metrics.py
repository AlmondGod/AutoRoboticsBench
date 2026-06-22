"""Minimal metric helpers."""

from __future__ import annotations


def success_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    successes = sum(1 for item in results if bool(item.get("success")))
    return successes / len(results)
