from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import prepare


# Autoresearch agents should edit this file, not prepare.py or eval/.
METHOD = 'bc'
CHANGE = 'BC wrist dropout'
STRATEGY_KIND = 'bc'
TRAIN_STEPS = 1_000_000
BATCH_SIZE = 64
LR = 0.0003
N_EMBD = 256
LOSS = 'mse'
CHUNK_DECAY = 1.0
IMAGE_NOISE = 0.0
ACTION_NOISE = 0.0
HISTORY_DROPOUT = 0.0
WRIST_DROPOUT = 0.35
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.0
VIDEO_AUX = False
TOKENIZER_CODEBOOK = 128
TOKENIZER_EMBD = 64
WORLD_LAYERS = 4
WORLD_HEADS = 4
WORLD_EMBD = 128
LOG_EVERY = 250
DEVICE = "auto"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/libero/train_py")
    parser.add_argument("--max-train-seconds", type=float, default=prepare.TRAIN_TIME_SECONDS)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    aux_metrics = {}
    if VIDEO_AUX:
        aux_metrics = _run_video_aux(out_dir, args.max_train_seconds)
    remaining = max(1.0, args.max_train_seconds - (time.time() - started))
    _run(
        [
            "python",
            "train/train_policy.py",
            "--data",
            str(prepare.PAIRED_SHARD),
            "--out-dir",
            str(out_dir),
            "--method",
            METHOD,
            "--steps",
            str(TRAIN_STEPS),
            "--batch-size",
            str(BATCH_SIZE),
            "--lr",
            str(LR),
            "--n-embd",
            str(N_EMBD),
            "--loss",
            LOSS,
            "--chunk-decay",
            str(CHUNK_DECAY),
            "--image-noise",
            str(IMAGE_NOISE),
            "--action-noise",
            str(ACTION_NOISE),
            "--history-dropout",
            str(HISTORY_DROPOUT),
            "--wrist-dropout",
            str(WRIST_DROPOUT),
            "--weight-decay",
            str(WEIGHT_DECAY),
            "--grad-clip",
            str(GRAD_CLIP),
            "--log-every",
            str(LOG_EVERY),
            "--device",
            DEVICE,
            "--max-train-seconds",
            str(remaining),
        ]
    )

    metrics = json.loads((out_dir / "metrics.json").read_text())
    metrics["change"] = CHANGE
    metrics["commit"] = _git_commit()
    metrics["max_train_seconds"] = args.max_train_seconds
    metrics["strategy_kind"] = STRATEGY_KIND
    metrics.update(aux_metrics)
    if not args.skip_eval:
        success = prepare.eval_policy(out_dir / "policy.pt", out_dir / "libero_success.json")
        metrics["success_rate"] = success["success_rate"]
        metrics["success_episodes"] = success["episodes"]
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _run(cmd: list[str]) -> None:
    if cmd and cmd[0] == "python":
        cmd = [sys.executable, *cmd[1:]]
    subprocess.run(cmd, check=True)


def _run_video_aux(out_dir: Path, max_train_seconds: float) -> dict:
    tok_dir = out_dir / "tokenizer"
    world_dir = out_dir / "world_model"
    tok_budget = max(1.0, 0.2 * max_train_seconds)
    world_budget = max(1.0, 0.2 * max_train_seconds)
    _run(
        [
            "python",
            "train/train_tokenizer.py",
            "--data",
            str(prepare.VIDEO_SHARD),
            "--out-dir",
            str(tok_dir),
            "--steps",
            str(TRAIN_STEPS),
            "--batch-size",
            str(BATCH_SIZE),
            "--codebook-size",
            str(TOKENIZER_CODEBOOK),
            "--embed-dim",
            str(TOKENIZER_EMBD),
            "--device",
            DEVICE,
            "--max-train-seconds",
            str(tok_budget),
            "--log-every",
            str(LOG_EVERY),
        ]
    )
    _run(
        [
            "python",
            "train/train_world_model.py",
            "--data",
            str(prepare.VIDEO_SHARD),
            "--tokenizer",
            str(tok_dir / "tokenizer.pt"),
            "--out-dir",
            str(world_dir),
            "--steps",
            str(TRAIN_STEPS),
            "--batch-size",
            str(BATCH_SIZE),
            "--layers",
            str(WORLD_LAYERS),
            "--heads",
            str(WORLD_HEADS),
            "--embd",
            str(WORLD_EMBD),
            "--device",
            DEVICE,
            "--max-train-seconds",
            str(world_budget),
            "--log-every",
            str(LOG_EVERY),
        ]
    )
    tok_metrics = json.loads((tok_dir / "metrics.json").read_text())
    world_metrics = json.loads((world_dir / "metrics.json").read_text())
    return {
        "video_loss": tok_metrics.get("video_loss"),
        "tokenizer_loss": tok_metrics.get("tokenizer_loss"),
        "val_video_nll": world_metrics.get("val_video_nll"),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except subprocess.SubprocessError:
        return "unknown"


if __name__ == "__main__":
    main()
