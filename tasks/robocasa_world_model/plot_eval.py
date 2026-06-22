from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RoboCasa reward-model correlation eval.")
    parser.add_argument("--eval", default="runs/autorobobench/robocasa_reward_model/quick_real/eval_correlation.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    eval_path = Path(args.eval)
    payload = json.loads(eval_path.read_text())
    out = Path(args.out) if args.out else eval_path.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(payload, out)
    print(out)


def plot(payload: dict, out: Path) -> None:
    corr = payload["policy_correlation"]
    policies = [row for row in corr["policies"] if row.get("real_success_rate") is not None and row.get("predicted_success") is not None]
    if not policies:
        raise ValueError("no valid policies with real_success_rate and predicted_success")

    names = [short_name(row["name"]) for row in policies]
    real = np.asarray([float(row["real_success_rate"]) for row in policies], dtype=np.float64)
    pred = np.asarray([float(row["predicted_success"]) for row in policies], dtype=np.float64)
    ood = np.asarray([bool(row.get("ood", False)) for row in policies], dtype=bool)
    metrics = payload["transition_metrics"]

    fig = plt.figure(figsize=(12, 7), dpi=160)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.4], height_ratios=[1.0, 0.8], wspace=0.28, hspace=0.34)
    ax_scatter = fig.add_subplot(grid[:, 0])
    ax_bar = fig.add_subplot(grid[0, 1])
    ax_metrics = fig.add_subplot(grid[1, 1])

    colors = np.where(ood, "#d95f02", "#1b9e77")
    ax_scatter.scatter(real, pred, s=86, c=colors, edgecolor="#222222", linewidth=0.8)
    for idx, name in enumerate(names):
        ax_scatter.annotate(name, (real[idx], pred[idx]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    if len(real) >= 2:
        x0, x1 = min(real) - 0.01, max(real) + 0.035
        slope, intercept = np.polyfit(real, pred, 1)
        xs = np.linspace(x0, x1, 64)
        ax_scatter.plot(xs, slope * xs + intercept, color="#4b4b4b", linewidth=1.4)
        ax_scatter.set_xlim(x0, x1)
    ax_scatter.set_title("Policy Score Correlation", fontsize=12, fontweight="bold")
    ax_scatter.set_xlabel("Real RoboCasa success rate")
    ax_scatter.set_ylabel("Reward-model predicted success")
    ax_scatter.grid(True, color="#dddddd", linewidth=0.8)
    ax_scatter.text(
        0.02,
        0.98,
        f"Benchmark {payload.get('reward_model_benchmark_score', 0.0):.3f}\nPearson {corr.get('pearson'):.3f}\nSpearman {corr.get('spearman'):.3f}\nN={corr.get('valid_policy_count')}",
        transform=ax_scatter.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    order = np.argsort(real)
    x = np.arange(len(policies))
    width = 0.38
    ax_bar.bar(x - width / 2, real[order], width, label="real success", color="#7570b3")
    ax_bar.bar(x + width / 2, pred[order], width, label="reward predicted", color="#66a61e")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([names[i] for i in order], rotation=18, ha="right", fontsize=8)
    ax_bar.set_title("Per-Policy Scores", fontsize=12, fontweight="bold")
    ax_bar.legend(frameon=False, fontsize=9)
    ax_bar.grid(True, axis="y", color="#dddddd", linewidth=0.8)

    metric_names = ["next_progress_mse", "success_bce"]
    metric_values = [float(metrics[name]) for name in metric_names]
    ax_metrics.barh(np.arange(len(metric_names)), metric_values, color=["#1b9e77", "#66a61e", "#e6ab02", "#d95f02"])
    ax_metrics.set_yticks(np.arange(len(metric_names)))
    ax_metrics.set_yticklabels([label.replace("_", " ") for label in metric_names], fontsize=9)
    ax_metrics.invert_yaxis()
    ax_metrics.set_title("Heldout Transition Metrics", fontsize=12, fontweight="bold")
    ax_metrics.grid(True, axis="x", color="#dddddd", linewidth=0.8)
    for idx, value in enumerate(metric_values):
        ax_metrics.text(value, idx, f" {value:.4f}", va="center", fontsize=8)

    fig.suptitle("RoboCasa Reward Model Eval", fontsize=14, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def short_name(name: str) -> str:
    replacements = {
        "autoresearch_clip_recede4_open_only": "recede4_open",
        "autoresearch_full_history_act_seed0": "full_history",
        "autoresearch_": "",
        "_seed0": "",
        "_5min": "",
        "robocasa_bc5_": "",
    }
    out = str(name)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


if __name__ == "__main__":
    main()
