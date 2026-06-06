from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNS_DIR = Path("runs")
LEDGER_PATH = RUNS_DIR / "research_log.jsonl"


def utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def git_diff_summary() -> str:
    try:
        return subprocess.check_output(
            ["git", "diff", "--stat"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_ledger(entry: dict[str, Any], ledger_path: Path = LEDGER_PATH) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def mark_ledger_decision(run_id: str, accepted: bool, ledger_path: Path = LEDGER_PATH) -> None:
    if not ledger_path.exists():
        return
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    for row in rows:
        if row.get("run_id") == run_id:
            row["accepted"] = accepted
    ledger_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
