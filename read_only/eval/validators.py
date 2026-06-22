"""Submission validation helpers."""

from __future__ import annotations

from pathlib import Path


def validate_submission(path: str | Path) -> tuple[bool, list[str]]:
    submission_path = Path(path)
    if not submission_path.exists():
        return False, ["submission_missing"]
    if not submission_path.is_dir():
        return False, ["submission_not_directory"]
    return True, []
