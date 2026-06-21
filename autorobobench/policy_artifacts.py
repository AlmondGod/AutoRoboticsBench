from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/autorobobench/pretrained_policies.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="List and verify registered pretrained policy artifacts.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--name", default="")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text())
    policies = list(payload.get("policies", []))
    if args.name:
        policies = [policy for policy in policies if str(policy.get("name")) == str(args.name)]
        if not policies:
            raise SystemExit(f"policy not found: {args.name}")

    rows = []
    ok = True
    for policy in policies:
        row = _policy_row(policy, verify=bool(args.verify))
        rows.append(row)
        ok = ok and bool(row.get("ok", True))

    print(json.dumps({"manifest": str(manifest_path), "policies": rows}, indent=2, sort_keys=True))
    if args.verify and not ok:
        raise SystemExit(1)


def _policy_row(policy: dict[str, Any], *, verify: bool) -> dict[str, Any]:
    row = {
        "name": policy.get("name"),
        "checkpoint": policy.get("checkpoint"),
        "policy_kind": policy.get("policy_kind"),
        "real_success_rate": policy.get("real_success_rate"),
        "wm_policy_improvement_supported": policy.get("wm_policy_improvement_supported"),
    }
    checkpoint = policy.get("checkpoint")
    if not checkpoint:
        row["ok"] = False
        row["status"] = "missing_checkpoint"
        return row
    if str(policy.get("artifact_status", "")) == "external_or_local_runs":
        row["ok"] = True
        row["status"] = "external_or_local_runs"
        row["verified"] = False
        return row

    path = _resolve_repo_path(str(checkpoint))
    row["path"] = str(path)
    row["exists"] = path.exists()
    if not path.exists():
        row["ok"] = False
        return row

    expected_size = policy.get("size_bytes")
    expected_sha = policy.get("sha256")
    actual_size = path.stat().st_size
    row["size_bytes"] = actual_size
    row["size_ok"] = expected_size is None or int(expected_size) == int(actual_size)
    if expected_sha and verify:
        actual_sha = _sha256(path)
        row["sha256"] = actual_sha
        row["sha256_ok"] = str(expected_sha) == actual_sha
    elif expected_sha:
        row["sha256_ok"] = True
        row["sha256_verified"] = False
    else:
        row["sha256_ok"] = not verify
        row["sha256_missing"] = True
    row["ok"] = bool(row["size_ok"] and row["sha256_ok"])
    return row


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
