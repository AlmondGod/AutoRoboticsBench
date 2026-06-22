#!/usr/bin/env python3
"""Check that RoboCasa BC wrapper tasks share the intended policy architecture."""

from __future__ import annotations

import ast
import json
from pathlib import Path


TASKS = [
    "robocasa_long_horizon",
    "robocasa_faucet_peak",
    "robocasa_stand_mixer_peak",
]

ARCH_FLAGS = [
    "--policy-kind",
    "--chunk-horizon",
    "--width",
    "--dropout",
    "--transformer-depth",
    "--action-depth",
    "--heads",
]

BC5_DEFAULTS = {
    "--policy-kind": "bc",
    "--chunk-horizon": "16",
    "--width": "256",
    "--dropout": "0.05",
    "--transformer-depth": "3",
    "--action-depth": "3",
    "--heads": "4",
}


def parse_wrapper_defaults(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defaults: dict[str, str] = {}
    constants: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        constants[target.id] = str(ast.literal_eval(node.value))
                    except (ValueError, SyntaxError):
                        pass
                if isinstance(target, ast.Name) and target.id == "DEFAULT_ARGS":
                    value = ast.literal_eval(node.value)
                    return {str(key): str(val) for key, val in value.items()}
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_default"
            and len(node.args) >= 2
        ):
            flag = ast.literal_eval(node.args[0])
            try:
                value = ast.literal_eval(node.args[1])
            except (ValueError, SyntaxError):
                if isinstance(node.args[1], ast.Name):
                    value = constants.get(node.args[1].id)
                else:
                    value = None
            if value is None:
                continue
            defaults[str(flag)] = str(value)
    return defaults


def source_contains(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


def architecture_defaults(wrapper_defaults: dict[str, str]) -> dict[str, str]:
    merged = dict(BC5_DEFAULTS)
    merged.update({key: value for key, value in wrapper_defaults.items() if key in ARCH_FLAGS})
    return {key: merged[key] for key in ARCH_FLAGS}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    reference = None
    report = {}
    failures = []

    for task in TASKS:
        train_path = repo_root / "tasks" / task / "train.py"
        inference_path = repo_root / "tasks" / task / "inference.py"
        defaults = parse_wrapper_defaults(train_path)
        arch = architecture_defaults(defaults)
        report[task] = {
            "uses_bc5_train": source_contains(train_path, "tasks.robocasa_bc5.train"),
            "uses_bc5_inference": source_contains(inference_path, "tasks.robocasa_bc5.inference"),
            "architecture": arch,
        }
        if not report[task]["uses_bc5_train"]:
            failures.append(f"{task} does not import tasks.robocasa_bc5.train")
        if not report[task]["uses_bc5_inference"]:
            failures.append(f"{task} does not import tasks.robocasa_bc5.inference")
        if reference is None:
            reference = arch
        elif arch != reference:
            failures.append(f"{task} architecture defaults differ from reference")

    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
