from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RoboCasa world-model train/eval artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-json", default="")
    parser.add_argument("--train-metrics", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "visualize"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = _resolve(args.eval_json, run_dir, "eval*.json")
    train_path = _resolve(args.train_metrics, run_dir, "train_metrics.json")
    eval_payload = _read_json(eval_path)
    train_payload = _read_json(train_path)

    if eval_payload:
        try:
            from tasks.robocasa_world_model.plot_eval import plot as plot_eval

            plot_eval(eval_payload, out_dir / "policy_correlation.png")
        except Exception as exc:  # visualization should not hide JSON summaries
            (out_dir / "policy_correlation_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    summary = _summary(run_dir, eval_path, train_path, eval_payload, train_payload)
    (out_dir / "world_model_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "world_model_summary.svg").write_text(_summary_svg(summary), encoding="utf-8")
    summary["outputs"] = {
        "summary_json": str(out_dir / "world_model_summary.json"),
        "summary_svg": str(out_dir / "world_model_summary.svg"),
        "policy_correlation_png": str(out_dir / "policy_correlation.png"),
    }
    (out_dir / "world_model_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _summary(
    run_dir: Path,
    eval_path: Path | None,
    train_path: Path | None,
    eval_payload: dict[str, Any],
    train_payload: dict[str, Any],
) -> dict[str, Any]:
    corr = eval_payload.get("policy_correlation", {}) if eval_payload else {}
    transition = eval_payload.get("transition_metrics", {}) if eval_payload else {}
    policies = corr.get("policies", []) if isinstance(corr, dict) else []
    policy_rows = [
        {
            "name": row.get("name", ""),
            "real_success_rate": row.get("real_success_rate"),
            "predicted_success": row.get("predicted_success"),
            "ood": bool(row.get("ood", False)),
            "trace_count": row.get("trace_count"),
        }
        for row in policies
        if isinstance(row, dict)
    ]
    return {
        "task": "robocasa_reward_model",
        "run_dir": str(run_dir),
        "eval_json": "" if eval_path is None else str(eval_path),
        "train_metrics": "" if train_path is None else str(train_path),
        "reward_model_benchmark_score": eval_payload.get("reward_model_benchmark_score"),
        "policy_ranking_score": eval_payload.get("policy_ranking_score"),
        "ood_ranking_score": eval_payload.get("ood_ranking_score"),
        "success_calibration_score": eval_payload.get("success_calibration_score"),
        "reward_score": eval_payload.get("reward_score"),
        "progress_score": eval_payload.get("progress_score") or eval_payload.get("reward_progress_score"),
        "transition_metrics": transition,
        "policy_rows": policy_rows,
        "final_val": train_payload.get("final_val", {}) if train_payload else {},
        "history_tail": train_payload.get("history", [])[-5:] if train_payload else [],
    }


def _summary_svg(summary: dict[str, Any]) -> str:
    rows = summary.get("policy_rows", [])
    width = 1050
    height = max(340, 180 + 36 * max(1, len(rows)))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#111">RoboCasa reward model</text>',
        f'<text x="24" y="68" font-family="Arial" font-size="14" fill="#333">benchmark={_fmt(summary.get("reward_model_benchmark_score"))} ranking={_fmt(summary.get("policy_ranking_score"))} ood={_fmt(summary.get("ood_ranking_score"))} calibration={_fmt(summary.get("success_calibration_score"))}</text>',
        f'<text x="24" y="94" font-family="Arial" font-size="12" fill="#666">{_esc(summary.get("eval_json", ""))}</text>',
    ]
    y0 = 136
    lines.append(f'<text x="24" y="{y0}" font-family="Arial" font-size="14" font-weight="700" fill="#111">Policy ranking</text>')
    for idx, row in enumerate(rows):
        y = y0 + 34 + idx * 36
        real = _as_float(row.get("real_success_rate"))
        pred = _as_float(row.get("predicted_success"))
        color = "#d97706" if row.get("ood") else "#2563eb"
        lines.append(f'<text x="24" y="{y}" font-family="Arial" font-size="12" fill="#111">{_esc(row.get("name", ""))}</text>')
        lines.append(f'<rect x="360" y="{y - 13}" width="220" height="12" fill="#e5e7eb"/>')
        lines.append(f'<rect x="360" y="{y - 13}" width="{220 * real:.1f}" height="12" fill="#16a34a"/>')
        lines.append(f'<rect x="610" y="{y - 13}" width="220" height="12" fill="#e5e7eb"/>')
        lines.append(f'<rect x="610" y="{y - 13}" width="{220 * pred:.1f}" height="12" fill="{color}"/>')
        lines.append(f'<text x="850" y="{y}" font-family="Arial" font-size="12" fill="#555">real {_fmt(real)} pred {_fmt(pred)}</text>')
    if not rows:
        lines.append(f'<text x="24" y="{y0 + 34}" font-family="Arial" font-size="12" fill="#666">No policy correlation rows found.</text>')
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
