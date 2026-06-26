"""Static replay-detection checks for submitted artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".py", ".json", ".txt", ".md"}
CHECKPOINT_SUFFIXES = {".pt", ".pth"}
FORBIDDEN_POLICY_TYPES = {
    "robocasa_bc5_trajectory_bank",
    "autorobobench_trajectory_bank",
    "trajectory_bank",
    "replay",
}
FORBIDDEN_CHECKPOINT_KEYS = {
    "actions",
    "episode_ids",
    "episodes",
    "eval_episode_ids",
    "frames",
    "images",
    "lengths",
    "observations",
    "reset_states",
    "states",
    "task_ids",
    "trajectory_bank",
    "trajectories",
}
FORBIDDEN_TEXT_PATTERNS = {
    "eval_episode_ids": re.compile(r"\beval_episode_ids\b"),
    "test_episodes": re.compile(r"\btest_episodes\b"),
    "heldout": re.compile(r"\bheldout\b", re.IGNORECASE),
    "eval_episode_env": re.compile(r"\bAUTOROBOBENCH_EVAL_EPISODE_ID\b"),
    "reset_state_index": re.compile(r"\breset_state_index\b"),
    "data_autorobobench": re.compile(r"data/autorobobench"),
    "data_robocasa": re.compile(r"data/(robocasa|robocasa5)"),
    "workspace_data": re.compile(r"/workspace/(repo/)?data/"),
}


def detect_replay(submission_path: str | Path) -> list[str]:
    root = Path(submission_path)
    flags: list[str] = []
    if not root.exists() or not root.is_dir():
        return flags

    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            flags.extend(_scan_text(path, rel))
        if suffix in CHECKPOINT_SUFFIXES:
            flags.extend(_scan_checkpoint(path, rel))
    return sorted(set(flags))


def _scan_text(path: Path, rel: str) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    flags = []
    for name, pattern in FORBIDDEN_TEXT_PATTERNS.items():
        if pattern.search(text):
            flags.append(f"replay_text:{name}:{rel}")
    return flags


def _scan_checkpoint(path: Path, rel: str) -> list[str]:
    try:
        import torch
    except ModuleNotFoundError:
        return []

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return [f"checkpoint_unreadable:{rel}:{type(exc).__name__}"]

    flags: list[str] = []
    _walk_payload(payload, rel, flags, depth=0)
    return flags


def _walk_payload(value: Any, rel: str, flags: list[str], depth: int) -> None:
    if depth > 5:
        return
    if isinstance(value, dict):
        policy_type = str(value.get("policy_type", "")).lower()
        mode = str(value.get("mode", "")).lower()
        if policy_type in FORBIDDEN_POLICY_TYPES or mode in FORBIDDEN_POLICY_TYPES:
            flags.append(f"replay_checkpoint:forbidden_policy_type:{rel}:{policy_type or mode}")
        for key, child in value.items():
            key_text = str(key)
            key_leaf = key_text.rsplit(".", 1)[-1].lower()
            if key_leaf in FORBIDDEN_CHECKPOINT_KEYS:
                flags.append(f"replay_checkpoint:forbidden_key:{rel}:{key_text}")
            _walk_payload(child, rel, flags, depth + 1)
    elif isinstance(value, (list, tuple)):
        for child in list(value)[:8]:
            _walk_payload(child, rel, flags, depth + 1)
