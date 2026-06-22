#!/usr/bin/env python3
"""Plot simple RoboAutoresearch scaling charts."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing input CSV: {path}. Run scripts/aggregate_runs.py first.")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_value(row: dict[str, str]) -> float | None:
    return to_float(row.get("normalized_score")) or to_float(row.get("score"))


def label_value(row: dict[str, str]) -> str:
    return row.get("model") or row.get("agent") or "unknown"


def scatter_plot(rows: list[dict[str, str]], x_key: str, x_label: str, out_path: Path) -> bool:
    points = []
    for row in rows:
        x = to_float(row.get(x_key))
        y = score_value(row)
        if x is None or y is None:
            continue
        if x_key == "wall_seconds":
            x = x / 3600.0
        points.append((x, y, label_value(row)))

    if not points:
        return False

    labels = sorted({point[2] for point in points})
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(idx % 10) for idx, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for label in labels:
        xs = [point[0] for point in points if point[2] == label]
        ys = [point[1] for point in points if point[2] == label]
        ax.scatter(xs, ys, label=label, color=colors[label], alpha=0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalized score" if any(row.get("normalized_score") for row in rows) else "Score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def leaderboard_plot(rows: list[dict[str, str]], out_path: Path) -> bool:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        y = score_value(row)
        if y is None:
            continue
        grouped[label_value(row)].append(y)

    if not grouped:
        return False

    items = sorted(
        ((label, sum(scores) / len(scores)) for label, scores in grouped.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    labels = [item[0] for item in items]
    scores = [item[1] for item in items]

    fig_width = max(7, min(14, 0.55 * len(labels) + 3))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.bar(labels, scores, color="#4C78A8")
    ax.set_ylabel("Mean normalized score" if any(row.get("normalized_score") for row in rows) else "Mean score")
    ax.set_xlabel("Model / agent")
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def has_token_data(rows: list[dict[str, str]]) -> bool:
    return any(to_float(row.get("total_tokens")) is not None for row in rows)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    analysis_dir = repo_root / "analysis"
    rows = read_csv(analysis_dir / "runs.csv")

    made_time = scatter_plot(
        rows,
        "wall_seconds",
        "Wall time (hours)",
        analysis_dir / "time_vs_score.png",
    )
    if has_token_data(rows):
        scatter_plot(
            rows,
            "total_tokens",
            "Total tokens",
            analysis_dir / "tokens_vs_score.png",
        )
    made_leaderboard = leaderboard_plot(rows, analysis_dir / "leaderboard.png")

    if made_time:
        print(f"Wrote {analysis_dir / 'time_vs_score.png'}")
    if has_token_data(rows):
        print(f"Wrote {analysis_dir / 'tokens_vs_score.png'}")
    if made_leaderboard:
        print(f"Wrote {analysis_dir / 'leaderboard.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
