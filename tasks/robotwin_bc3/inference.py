from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


def load_policy(checkpoint: str, device: str = "cuda"):
    """Load a LeRobot policy for ad hoc Python inference.

    RoboTwin BC-3 evaluation uses `lerobot-eval` directly because the simulator
    supplies the language-conditioned observations and task strings.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = device
    policy = make_policy(cfg=cfg)
    policy.eval()
    return policy


def act(policy, obs: dict, task: dict):
    raise NotImplementedError(
        "RoboTwin BC-3 policies should be evaluated through tasks/robotwin_bc3/eval.py "
        "or lerobot-eval so the RoboTwin env can inject task language correctly."
    )
