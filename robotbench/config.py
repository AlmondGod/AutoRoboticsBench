from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).parent / "tasks"


@dataclass(frozen=True)
class TaskConfig:
    name: str
    horizon: int
    success_tolerance: float
    action_limit: float
    train: dict[str, Any]
    eval: dict[str, Any]


def load_task(name: str) -> TaskConfig:
    path = TASK_DIR / f"{name}.yaml"
    if not path.exists():
        known = ", ".join(sorted(p.stem for p in TASK_DIR.glob("*.yaml")))
        raise ValueError(f"unknown task '{name}'. Known tasks: {known}")

    raw = _load_task_file(path)
    return TaskConfig(
        name=raw["name"],
        horizon=int(raw["horizon"]),
        success_tolerance=float(raw["success_tolerance"]),
        action_limit=float(raw["action_limit"]),
        train=dict(raw["train"]),
        eval=dict(raw["eval"]),
    )


def _load_task_file(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text())
    except ModuleNotFoundError:
        return _parse_simple_yaml(path.read_text())


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, value = raw_line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value
