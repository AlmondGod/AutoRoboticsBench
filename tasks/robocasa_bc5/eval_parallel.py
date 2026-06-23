from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


@dataclass
class WorkerProc:
    worker_idx: int
    process: subprocess.Popen
    result_path: Path
    log_path: Path
    log_handle: TextIO


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard RoboCasa BC-5 eval across parallel worker processes.")
    parser.add_argument("--checkpoint", "--policy", dest="checkpoint", required=True)
    parser.add_argument("--inference", default="tasks.robocasa_bc5.inference")
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--commit-steps", type=int, default=16)
    parser.add_argument("--eval-episodes-per-task", type=int, default=50)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--render-dir", default="")
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--render-episodes-per-task", type=int, default=0)
    parser.add_argument("--render-width", type=int, default=768)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("AUTOROBOBENCH_EVAL_WORKERS", "28")))
    parser.add_argument("--worker-timeout-seconds", type=float, default=0.0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent / f"{out_path.stem}_parallel"
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text())
    split = json.loads(Path(args.split).read_text())
    task_aliases = set(args.task_alias)
    episode_rows, order = _episode_rows(split, task_aliases, int(args.eval_episodes_per_task))
    if not episode_rows:
        raise ValueError("no eval episodes selected")

    workers = max(1, min(int(args.workers), len(episode_rows)))
    shards = [episode_rows[idx::workers] for idx in range(workers)]
    worker_specs = []
    for worker_idx, rows in enumerate(shards):
        if not rows:
            continue
        shard_split = _shard_split(split, rows)
        shard_split_path = work_dir / f"split_worker_{worker_idx:03d}.json"
        shard_out = work_dir / f"result_worker_{worker_idx:03d}.json"
        shard_log = work_dir / f"worker_{worker_idx:03d}.log"
        shard_split_path.write_text(json.dumps(shard_split, indent=2, sort_keys=True) + "\n")
        worker_specs.append((worker_idx, rows, shard_split_path, shard_out, shard_log))

    start_time = time.monotonic()
    procs: list[WorkerProc] = []
    for worker_idx, rows, shard_split_path, shard_out, shard_log in worker_specs:
        cmd = [
            sys.executable,
            str(ROOT / "tasks/robocasa_bc5/eval.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--inference",
            str(args.inference),
            "--manifest",
            str(args.manifest),
            "--split",
            str(shard_split_path),
            "--out",
            str(shard_out),
            "--camera",
            str(args.camera),
            "--max-steps",
            str(int(args.max_steps)),
            "--commit-steps",
            str(int(args.commit_steps)),
            "--eval-episodes-per-task",
            "0",
            "--render-width",
            str(int(args.render_width)),
            "--render-height",
            str(int(args.render_height)),
            "--fps",
            str(int(args.fps)),
            "--device",
            str(args.device),
        ]
        if args.render_dir:
            render_dir = Path(args.render_dir) / f"worker_{worker_idx:03d}"
            cmd.extend(["--render-dir", str(render_dir)])
        if args.trace_dir:
            trace_dir = Path(args.trace_dir) / f"worker_{worker_idx:03d}"
            cmd.extend(["--trace-dir", str(trace_dir)])
        if int(args.render_episodes_per_task) > 0:
            cmd.extend(["--render-episodes-per-task", str(int(args.render_episodes_per_task))])
        with shard_log.open("w") as log:
            log.write(f"cmd: {' '.join(cmd)}\n")
        log_handle = shard_log.open("a")
        worker_env = os.environ.copy()
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            worker_env.setdefault(key, "1")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=worker_env,
        )
        procs.append(
            WorkerProc(
                worker_idx=worker_idx,
                process=proc,
                result_path=shard_out,
                log_path=shard_log,
                log_handle=log_handle,
            )
        )
        print(json.dumps({"worker": worker_idx, "episodes": len(rows), "pid": proc.pid, "log": str(shard_log)}), flush=True)

    deadline = None
    if float(args.worker_timeout_seconds) > 0:
        deadline = time.monotonic() + float(args.worker_timeout_seconds)
    failures = []
    for worker in procs:
        timeout = None
        if deadline is not None:
            timeout = max(0.0, deadline - time.monotonic())
        timed_out = False
        try:
            code = worker.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            worker.process.kill()
            code = worker.process.wait(timeout=30)
            timed_out = True
            failures.append((worker.worker_idx, f"timeout exit={code}", worker.log_path))
        finally:
            worker.log_handle.close()
        if code != 0 and not timed_out:
            failures.append((worker.worker_idx, f"exit={code}", worker.log_path))
        elif not worker.result_path.exists():
            failures.append((worker.worker_idx, "missing result", worker.log_path))
        print(json.dumps({"worker": worker.worker_idx, "exit": int(code), "result": str(worker.result_path)}), flush=True)

    if failures:
        for worker_idx, reason, shard_log in failures:
            print(f"worker {worker_idx} failed: {reason}; log={shard_log}", file=sys.stderr)
            if shard_log.exists():
                print(shard_log.read_text()[-4000:], file=sys.stderr)
        raise SystemExit(1)

    payload = _merge_results(
        args=args,
        manifest=manifest,
        split=split,
        shard_results=[worker.result_path for worker in procs],
        order=order,
        workers=len(procs),
        elapsed_seconds=time.monotonic() - start_time,
    )
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def _episode_rows(split: dict, task_aliases: set[str], eval_episodes_per_task: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    rows = []
    order = {}
    ordinal = 0
    for task_idx, split_task in enumerate(split["tasks"]):
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        episode_ids = list(split_task["eval_episode_ids"])
        if eval_episodes_per_task > 0:
            episode_ids = episode_ids[:eval_episodes_per_task]
        for local_idx, episode_id in enumerate(episode_ids):
            row = {
                "task_idx": int(task_idx),
                "task_alias": alias,
                "task_id": int(split_task["task_id"]),
                "episode_id": int(episode_id),
                "local_idx": int(local_idx),
            }
            rows.append(row)
            order[(int(split_task["task_id"]), int(episode_id))] = ordinal
            ordinal += 1
    return rows, order


def _shard_split(split: dict, rows: list[dict]) -> dict:
    out = deepcopy(split)
    by_task: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task_idx"])].append(int(row["episode_id"]))
    for task_idx, split_task in enumerate(out["tasks"]):
        split_task["eval_episode_ids"] = by_task.get(int(task_idx), [])
    return out


def _merge_results(
    *,
    args: argparse.Namespace,
    manifest: dict,
    split: dict,
    shard_results: list[Path],
    order: dict[tuple[int, int], int],
    workers: int,
    elapsed_seconds: float,
) -> dict:
    details = []
    worker_payloads = []
    for result_path in shard_results:
        payload = json.loads(result_path.read_text())
        worker_payloads.append(
            {
                "path": str(result_path),
                "episodes": int(payload.get("episodes", 0)),
                "successes": int(payload.get("successes", 0)),
                "success_rate": float(payload.get("success_rate", 0.0)),
            }
        )
        details.extend(payload.get("details", []))
    details.sort(key=lambda row: order.get((int(row["task_id"]), int(row["episode_id"])), 10**12))

    per_task = {}
    by_alias: dict[str, list[dict]] = defaultdict(list)
    for row in details:
        by_alias[str(row["task_alias"])].append(row)
    for alias, rows in by_alias.items():
        successes = sum(int(bool(row["success"])) for row in rows)
        per_task[alias] = {
            "episodes": len(rows),
            "successes": int(successes),
            "success_rate": successes / max(1, len(rows)),
        }
    successes = sum(int(bool(row["success"])) for row in details)
    return {
        "track": "robocasa_bc5",
        "checkpoint": str(args.checkpoint),
        "inference": str(args.inference),
        "manifest": str(args.manifest),
        "split": str(args.split),
        "episodes": len(details),
        "successes": int(successes),
        "success_rate": successes / max(1, len(details)),
        "commit_steps": int(args.commit_steps),
        "max_steps": int(args.max_steps),
        "requested_eval_episodes_per_task": int(args.eval_episodes_per_task),
        "available_eval_episodes_per_task": {
            str(task["alias"]): len(task.get("eval_episode_ids", []))
            for task in split.get("tasks", [])
            if not args.task_alias or str(task["alias"]) in set(args.task_alias)
        },
        "per_task": per_task,
        "details": details,
        "parallel_eval": {
            "workers": int(workers),
            "elapsed_seconds": float(elapsed_seconds),
            "worker_results": worker_payloads,
        },
        "split_track": split.get("track"),
        "manifest_track": manifest.get("track"),
    }


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "glfw")
    main()
