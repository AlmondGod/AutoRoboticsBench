from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyTracker:
    action_limit: float

    def reset(self) -> None:
        self.joint_limit_violations = 0
        self.torque_limit_violations = 0
        self.catastrophe = False

    def observe(self, raw_action: np.ndarray, clipped_action: np.ndarray) -> None:
        if np.any(np.abs(raw_action) > self.action_limit * 1.01):
            self.torque_limit_violations += 1
        if np.any(~np.isfinite(raw_action)):
            self.catastrophe = True
        if np.any(np.abs(clipped_action) > self.action_limit * 1.01):
            self.joint_limit_violations += 1

    def snapshot(self) -> dict[str, int | bool]:
        return {
            "joint_limit_violations": self.joint_limit_violations,
            "torque_limit_violations": self.torque_limit_violations,
            "catastrophe": self.catastrophe,
        }
