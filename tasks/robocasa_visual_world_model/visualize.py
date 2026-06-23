from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RoboCasa visual world-model artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--eval-json", default="")
    parser.add_argument("--train-metrics", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--rollout", action="store_true", help="Also write a predicted-vs-actual rollout GIF.")
    parser.add_argument("--mode", choices=["teacher_forced", "closed_loop"], default="closed_loop")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "visualize"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = _resolve(args.eval_json, run_dir, "eval*.json")
    train_path = _resolve(args.train_metrics, run_dir, "train_metrics.json")
    eval_payload = _read_json(eval_path)
    train_payload = _read_json(train_path)

    rollout = {}
    if args.rollout:
        checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "policy_best.pt"
        gif_path = out_dir / f"rollout_{args.mode}.gif"
        cmd = [
            sys.executable,
            "tasks/robocasa_visual_world_model/visualize_rollout.py",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(gif_path),
            "--mode",
            str(args.mode),
            "--max-steps",
            str(int(args.max_steps)),
            "--device",
            str(args.device),
        ]
        subprocess.run(cmd, check=True)
        rollout = {"cmd": " ".join(cmd), "gif": str(gif_path), "preview_png": str(gif_path.with_suffix(".png"))}

    summary = _summary(run_dir, eval_path, train_path, eval_payload, train_payload, rollout)
    (out_dir / "visual_world_model_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "visual_world_model_summary.svg").write_text(_summary_svg(summary), encoding="utf-8")
    summary["outputs"] = {
        "summary_json": str(out_dir / "visual_world_model_summary.json"),
        "summary_svg": str(out_dir / "visual_world_model_summary.svg"),
        **({"rollout_gif": rollout["gif"]} if rollout else {}),
    }
    (out_dir / "visual_world_model_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _summary(
    run_dir: Path,
    eval_path: Path | None,
    train_path: Path | None,
    eval_payload: dict[str, Any],
    train_payload: dict[str, Any],
    rollout: dict[str, Any],
) -> dict[str, Any]:
    metrics = eval_payload.get("visual_transition_metrics", {}) if eval_payload else {}
    return {
        "task": "robocasa_visual_world_model",
        "run_dir": str(run_dir),
        "eval_json": "" if eval_path is None else str(eval_path),
        "train_metrics": "" if train_path is None else str(train_path),
        "visual_world_model_score": eval_payload.get("visual_world_model_score"),
        "eval_correlation_score": eval_payload.get("eval_correlation_score"),
        "policy_score_pearson": eval_payload.get("policy_score_pearson"),
        "policy_score_spearman": eval_payload.get("policy_score_spearman"),
        "visual_perceptual_score": eval_payload.get("visual_perceptual_score"),
        "visual_reconstruction_score": eval_payload.get("visual_reconstruction_score"),
        "next_state_score": eval_payload.get("next_state_score"),
        "progress_score": eval_payload.get("progress_score") or eval_payload.get("reward_progress_score"),
        "success_score": eval_payload.get("success_score"),
        "visual_transition_metrics": metrics,
        "final_val": train_payload.get("final_val", {}) if train_payload else {},
        "history_tail": train_payload.get("history", [])[-5:] if train_payload else [],
        "rollout": rollout,
    }


def _summary_svg(summary: dict[str, Any]) -> str:
    bars = [
        ("visual score", summary.get("visual_world_model_score")),
        ("eval corr", summary.get("eval_correlation_score")),
        ("pixel mse", summary.get("visual_reconstruction_score")),
        ("next state", summary.get("next_state_score")),
        ("progress", summary.get("progress_score")),
        ("success", summary.get("success_score")),
    ]
    width, height = 900, 410
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#111">Visual world model</text>',
        f'<text x="24" y="66" font-family="Arial" font-size="12" fill="#666">{_esc(summary.get("eval_json", ""))}</text>',
    ]
    for idx, (name, value) in enumerate(bars):
        y = 112 + idx * 42
        score = _as_float(value)
        lines.append(f'<text x="24" y="{y}" font-family="Arial" font-size="14" fill="#111">{_esc(name)}</text>')
        lines.append(f'<rect x="210" y="{y - 15}" width="520" height="18" fill="#e5e7eb"/>')
        lines.append(f'<rect x="210" y="{y - 15}" width="{520 * score:.1f}" height="18" fill="#7c3aed"/>')
        lines.append(f'<text x="750" y="{y}" font-family="Arial" font-size="13" fill="#333">{_fmt(value)}</text>')
    if summary.get("rollout"):
        lines.append(f'<text x="24" y="380" font-family="Arial" font-size="13" fill="#333">rollout: {_esc(summary["rollout"].get("gif", ""))}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _resolve(value: str, run_dir: Path, pattern: str) -> Path | None:
    if value:
        path = Path(value)
        return path if path.exists() else None
    exact = run_dir / pattern
    if "*" not in pattern and exact.exists():
        return exact
    candidates = sorted(run_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _as_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
