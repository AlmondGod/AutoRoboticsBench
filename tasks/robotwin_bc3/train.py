from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robotwin_bc3.common import (
    DEFAULT_FINETUNE_SECONDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRETRAINED_POLICY,
    DEFAULT_SPLIT,
    RENAME_MAP,
    latest_checkpoint,
    load_split,
    require_command,
    split_train_episode_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA on RoboTwin BC-3 demonstrations.")
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--pretrained-policy", default=DEFAULT_PRETRAINED_POLICY)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "smolvla_robotwin_5min"))
    parser.add_argument("--out", default="")
    parser.add_argument("--max-train-seconds", type=int, default=DEFAULT_FINETUNE_SECONDS)
    parser.add_argument("--demos-per-task", type=int, default=50)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-freq", type=int, default=25)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    split = load_split(args.split)
    repo_id = str(args.repo_id or split["repo_id"])
    train_episodes = split_train_episode_ids(split, demos_per_task=int(args.demos_per_task))
    if not train_episodes:
        raise ValueError("no training episodes selected")

    output_dir = Path(args.output_dir)
    cmd = build_train_command(
        repo_id=repo_id,
        train_episodes=train_episodes,
        pretrained_policy=str(args.pretrained_policy),
        output_dir=output_dir,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        save_freq=int(args.save_freq),
        log_freq=int(args.log_freq),
        device=str(args.device),
    )
    if args.dry_run:
        result = {"returncode": None, "timed_out": False, "cmd": cmd, "checkpoint": None}
    else:
        require_command("lerobot-train")
        start = time.time()
        completed = run_with_optional_timeout(cmd, int(args.max_train_seconds))
        checkpoint = latest_checkpoint(output_dir)
        result = {
            "returncode": int(completed.returncode),
            "timed_out": completed.returncode == 124,
            "train_s": time.time() - start,
            "cmd": cmd,
            "checkpoint": None if checkpoint is None else str(checkpoint),
        }
        if completed.returncode not in (0, 124):
            raise subprocess.CalledProcessError(completed.returncode, cmd)
        if checkpoint is None:
            raise FileNotFoundError(
                f"training finished or timed out but no checkpoint was found under {output_dir / 'checkpoints'}"
            )

    payload = {
        "task": "robotwin_bc3",
        "repo_id": repo_id,
        "split": str(args.split),
        "pretrained_policy": str(args.pretrained_policy),
        "output_dir": str(output_dir),
        "train_episode_count": len(train_episodes),
        "train_tasks": [task["robotwin_task"] for task in split["tasks"]],
        "max_train_seconds": int(args.max_train_seconds),
        "result": result,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text, end="")


def build_train_command(
    *,
    repo_id: str,
    train_episodes: list[int],
    pretrained_policy: str,
    output_dir: Path,
    steps: int,
    batch_size: int,
    num_workers: int,
    save_freq: int,
    log_freq: int,
    device: str,
) -> list[str]:
    episodes = "[" + ",".join(str(item) for item in train_episodes) + "]"
    return [
        "lerobot-train",
        f"--policy.path={pretrained_policy}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.episodes={episodes}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--num_workers={num_workers}",
        f"--save_freq={save_freq}",
        f"--log_freq={log_freq}",
        "--eval_freq=0",
        f"--output_dir={output_dir}",
        "--job_name=robotwin_bc3_smolvla",
        f"--policy.device={device}",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.train_state_proj=true",
        "--policy.num_expert_layers=-1",
        "--policy.num_vlm_layers=16",
        "--policy.load_vlm_weights=true",
        f"--rename_map={json.dumps(RENAME_MAP, separators=(',', ':'))}",
        "--wandb.enable=false",
    ]


def run_with_optional_timeout(cmd: list[str], max_train_seconds: int) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if max_train_seconds <= 0:
        print(json.dumps({"cmd": cmd}, indent=2), flush=True)
        return subprocess.run(cmd, check=False, text=True, env=env)
    timeout_bin = shutil.which("timeout")
    if timeout_bin:
        wrapped = [timeout_bin, "--preserve-status", "--signal=TERM", str(max_train_seconds), *cmd]
        print(json.dumps({"cmd": wrapped}, indent=2), flush=True)
        completed = subprocess.run(wrapped, check=False, text=True, env=env)
        if completed.returncode == 143:
            completed.returncode = 124
        return completed
    print(json.dumps({"cmd": cmd, "python_timeout_s": max_train_seconds}, indent=2), flush=True)
    try:
        return subprocess.run(cmd, check=False, text=True, env=env, timeout=max_train_seconds)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, exc.output, exc.stderr)


if __name__ == "__main__":
    main()
