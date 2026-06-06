from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="runs/research_log.jsonl")
    parser.add_argument("--out", default="runs/progress.svg")
    args = parser.parse_args()
    plot_progress(Path(args.log), Path(args.out))


def plot_progress(log_path: Path, out_path: Path) -> None:
    if not log_path.exists():
        raise FileNotFoundError(f"missing research log: {log_path}")
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty research log: {log_path}")

    x = list(range(len(rows)))
    scores = [float(row["eval_score"]) for row in rows]
    accepted = [row.get("accepted") for row in rows]
    labels = [
        f"{row.get('git_commit', 'unknown')[:7]}\n{row.get('change_note', '')[:28]}"
        for row in rows
    ]

    if out_path.suffix.lower() == ".png":
        try:
            _plot_png(x=x, scores=scores, labels=labels, accepted=accepted, out_path=out_path)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PNG output requires matplotlib. Use --out runs/progress.svg "
                "or install the optional plot dependency."
            ) from exc
    else:
        _plot_svg(scores=scores, labels=labels, accepted=accepted, out_path=out_path)
    print(out_path)


def _plot_png(
    x: list[int],
    scores: list[float],
    labels: list[str],
    accepted: list[bool | None],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = [_status_color(status) for status in accepted]
    plt.figure(figsize=(max(8, len(scores) * 1.2), 4.8))
    plt.plot(x, scores, marker="o", linewidth=2)
    plt.scatter(x, scores, c=colors, zorder=3)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Eval score")
    plt.xlabel("Run")
    plt.title("Autoresearch Progress")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)


def _plot_svg(
    scores: list[float],
    labels: list[str],
    accepted: list[bool | None],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = max(760, 150 * len(scores))
    height = 420
    margin_left = 70
    margin_right = 30
    margin_top = 35
    margin_bottom = 115
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        min_score -= 0.5
        max_score += 0.5
    padding = (max_score - min_score) * 0.08
    min_axis = min_score - padding
    max_axis = max_score + padding

    def point(idx: int, score: float) -> tuple[float, float]:
        x = margin_left + (plot_w * idx / max(1, len(scores) - 1))
        y = margin_top + plot_h * (1.0 - (score - min_axis) / (max_axis - min_axis))
        return x, y

    points = [point(idx, score) for idx, score in enumerate(scores)]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="26" font-family="Arial" font-size="18" font-weight="700">Autoresearch Progress</text>',
        '<text x="520" y="26" font-family="Arial" font-size="12"><tspan fill="#2ca02c">accepted</tspan> / <tspan fill="#d62728">rejected</tspan> / <tspan fill="#1f77b4">pending</tspan></text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222"/>',
    ]
    tick_count = 5
    for tick_idx in range(tick_count):
        value = min_axis + (max_axis - min_axis) * tick_idx / (tick_count - 1)
        y = margin_top + plot_h * (1.0 - (value - min_axis) / (max_axis - min_axis))
        parts.extend(
            [
                f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e5e5"/>',
                f'<text x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#444">{value:.2f}</text>',
            ]
        )
    parts.append(f'<polyline points="{polyline}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
    for idx, ((x, y), score, label, status) in enumerate(
        zip(points, scores, labels, accepted, strict=True)
    ):
        safe_label = _xml_escape(label.replace("\n", " "))
        color = _status_color(status)
        parts.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>',
                f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{score:.3f}</text>',
                f'<text x="{x:.1f}" y="{margin_top + plot_h + 22}" text-anchor="end" transform="rotate(-35 {x:.1f} {margin_top + plot_h + 22})" font-family="Arial" font-size="10">{safe_label}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="22" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 22 {margin_top + plot_h / 2:.1f})" font-family="Arial" font-size="12">Eval score</text>',
            "</svg>",
        ]
    )
    out_path.write_text("\n".join(parts) + "\n")


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _status_color(status: bool | None) -> str:
    if status is True:
        return "#2ca02c"
    if status is False:
        return "#d62728"
    return "#1f77b4"


if __name__ == "__main__":
    main()
