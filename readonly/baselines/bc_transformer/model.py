"""Placeholder BC Transformer model."""

from __future__ import annotations


class BCTransformer:
    def predict(self, observation: dict) -> dict:
        return {"action": "noop", "observation": observation}
