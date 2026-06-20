from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robotwin_bc3.common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRETRAINED_POLICY,
    DEFAULT_SPLIT,
    RENAME_MAP,
    load_split,
    parse_eval_info,
    require_command,
    split_eval_task_names,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SmolVLA policy on RoboTwin BC-3 tasks.")
    parser.add_argument("--checkpoint", "--policy", dest="checkpoint", default=DEFAULT_PRETRAINED_POLICY)
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "eval"))
    parser.add_argument("--eval-episodes-per-task", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    split = load_split(args.split)
    tasks = list(args.task) if args.task else split_eval_task_names(split)
    output_dir = Path(args.output_dir) / safe_name(str(args.checkpoint))
    cmd = build_eval_command(
        checkpoint=str(args.checkpoint),
        tasks=tasks,
        output_dir=output_dir,
        episodes_per_task=int(args.eval_episodes_per_task),
        batch_size=int(args.batch_size),
        device=str(args.device),
        seed=int(args.seed),
    )
    if args.dry_run:
        payload = {
            "task": "robotwin_bc3",
            "checkpoint": str(args.checkpoint),
            "cmd": cmd,
            "success_rate": None,
            "dry_run": True,
        }
    else:
        require_command("lerobot-eval")
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        start = time.time()
        print(json.dumps({"cmd": cmd}, indent=2), flush=True)
        completed = subprocess.run(cmd, check=False, text=True, env=env)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, cmd)
        eval_info = output_dir / "eval_info.json"
        if not eval_info.exists():
            raise FileNotFoundError(f"lerobot-eval did not write {eval_info}")
        parsed = parse_eval_info(eval_info)
        payload = {
            "task": "robotwin_bc3",
            "checkpoint": str(args.checkpoint),
            "split": str(args.split),
            "output_dir": str(output_dir),
            "eval_info": str(eval_info),
            "tasks": tasks,
            "eval_episodes_per_task": int(args.eval_episodes_per_task),
            "eval_s": time.time() - start,
            "success_rate": parsed["success_rate"],
            "pc_success": parsed["pc_success"],
            "episodes": parsed["episodes"],
            "per_task": parsed["per_task"],
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_eval_command(
    *,
    checkpoint: str,
    tasks: list[str],
    output_dir: Path,
    episodes_per_task: int,
    batch_size: int,
    device: str,
    seed: int,
) -> list[str]:
    return [
        "lerobot-eval",
        f"--policy.path={checkpoint}",
        "--env.type=robotwin",
        f"--env.task={','.join(tasks)}",
        f"--eval.batch_size={batch_size}",
        f"--eval.n_episodes={episodes_per_task}",
        f"--output_dir={output_dir}",
        "--job_name=robotwin_bc3_eval",
        f"--policy.device={device}",
        f"--seed={seed}",
        f"--rename_map={json.dumps(RENAME_MAP, separators=(',', ':'))}",
    ]


def safe_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
