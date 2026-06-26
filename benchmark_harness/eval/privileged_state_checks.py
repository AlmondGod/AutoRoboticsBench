"""Static checks for privileged simulator/eval access in submissions."""

from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN_IMPORT_PREFIXES = (
    "benchmark_harness",
    "robosuite.utils.binding_utils",
    "robosuite.utils.observables",
    "tasks.",
)
FORBIDDEN_NAME_PATTERNS = {
    "sim.data",
    "env.sim",
    "env.env",
    "get_state",
    "set_state",
    "check_success",
    "_check_success",
    "success_detector",
    "evaluate_submission",
}
FORBIDDEN_PATH_PATTERNS = {
    "/workspace/read_only",
    "benchmark_harness/eval",
}
ALLOWED_TASK_IMPORTS = {
    "tasks.robocasa_bc5.inference",
    "tasks.robocasa_bc1.inference",
    "tasks.robocasa_bc5_with_video.inference",
    "tasks.robocasa_long_horizon.inference",
    "tasks.robocasa_language_following.inference",
    "tasks.robocasa_world_model_posttraining.inference",
    "tasks.robocasa_offlinerl_posttraining.inference",
    "tasks.robocasa_visual_world_model.inference",
    "tasks.robocasa_world_model.inference",
}


def check_privileged_state_use(submission_path: str | Path) -> list[str]:
    root = Path(submission_path)
    flags: list[str] = []
    if not root.exists() or not root.is_dir():
        return flags

    for path in root.rglob("*.py"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        flags.extend(_scan_text(text, rel))
        flags.extend(_scan_ast(text, rel))
    return sorted(set(flags))


def _scan_text(text: str, rel: str) -> list[str]:
    flags = []
    lowered = text.lower()
    for pattern in FORBIDDEN_NAME_PATTERNS:
        if pattern.lower() in lowered:
            flags.append(f"privileged_text:{pattern}:{rel}")
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if pattern.lower() in lowered:
            flags.append(f"privileged_path:{pattern}:{rel}")
    return flags


def _scan_ast(text: str, rel: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [f"privileged_source_syntax_error:{rel}"]

    flags = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                flags.extend(_check_import(alias.name, rel))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            flags.extend(_check_import(module, rel))
        elif isinstance(node, ast.Attribute):
            attr = _attribute_name(node)
            if attr in {"sim.data", "env.sim", "env.env"}:
                flags.append(f"privileged_attr:{attr}:{rel}")
    return flags


def _check_import(module: str, rel: str) -> list[str]:
    if module in ALLOWED_TASK_IMPORTS:
        return []
    lowered = module.lower()
    for prefix in FORBIDDEN_IMPORT_PREFIXES:
        if lowered == prefix.rstrip(".") or lowered.startswith(prefix.lower()):
            return [f"privileged_import:{module}:{rel}"]
    return []


def _attribute_name(node: ast.Attribute) -> str:
    parts = [node.attr]
    cur = node.value
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))
