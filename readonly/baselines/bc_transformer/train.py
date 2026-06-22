#!/usr/bin/env python3
"""Placeholder baseline trainer."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    out_dir = Path("/workspace/output/final_submission")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps({"score": 0.1}, indent=2), encoding="utf-8")
    print(f"Wrote placeholder submission to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
