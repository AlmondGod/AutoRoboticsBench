"""Placeholder policy for the toy pickplace task."""

from __future__ import annotations


def act(observation: dict) -> dict:
    return {"action": "pickplace_noop", "observation": observation}
