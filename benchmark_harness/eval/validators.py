"""Submission validation helpers.

These checks intentionally stay conservative: they reject malformed submissions
and filesystem tricks before task-specific evaluation runs, while leaving policy
quality and task semantics to the real task evaluators.
"""

from __future__ import annotations

from pathlib import Path


MAX_FILES = 32
MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024
ALLOWED_SUFFIXES = {
    ".json",
    ".md",
    ".npz",
    ".pth",
    ".pt",
    ".py",
    ".safetensors",
    ".txt",
}
CHECKPOINT_NAMES = {
    "policy_best.pt",
    "policy.pt",
    "policy_last.pt",
    "checkpoint.pt",
    "model.pt",
    "world_model.pt",
}


def validate_submission(path: str | Path) -> tuple[bool, list[str]]:
    submission_path = Path(path)
    if not submission_path.exists():
        return False, ["submission_missing"]
    if not submission_path.is_dir():
        return False, ["submission_not_directory"]

    flags: list[str] = []
    files = []
    total_bytes = 0
    checkpoint_found = False

    for child in submission_path.rglob("*"):
        rel = child.relative_to(submission_path).as_posix()
        if child.is_symlink():
            flags.append(f"submission_symlink:{rel}")
            continue
        if child.is_dir():
            continue
        if not child.is_file():
            flags.append(f"submission_special_file:{rel}")
            continue
        files.append(child)
        suffix = child.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            flags.append(f"submission_disallowed_file_type:{rel}")
        if child.name.startswith("."):
            flags.append(f"submission_hidden_file:{rel}")
        try:
            size = child.stat().st_size
        except OSError:
            flags.append(f"submission_unreadable:{rel}")
            continue
        total_bytes += size
        if suffix == ".py" and size > MAX_SOURCE_BYTES:
            flags.append(f"submission_source_too_large:{rel}")
        if child.name in CHECKPOINT_NAMES or suffix in {".pt", ".pth", ".safetensors"}:
            checkpoint_found = True

    if not files:
        flags.append("submission_empty")
    if len(files) > MAX_FILES:
        flags.append(f"submission_too_many_files:{len(files)}")
    if total_bytes > MAX_TOTAL_BYTES:
        flags.append(f"submission_too_large:{total_bytes}")
    if not checkpoint_found:
        flags.append("checkpoint_missing")

    return not flags, flags
