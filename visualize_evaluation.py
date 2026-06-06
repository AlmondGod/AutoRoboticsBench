from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import research
from robotbench.config import load_task
from robotbench.envs import make_env
from robotbench.train import TrainBudget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach", choices=["reach", "push"])
    parser.add_argument("--backend", default="toy", choices=["toy", "mujoco", "aloha", "mobile_aloha_mock"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-seconds", type=float, default=60.0)
    parser.add_argument("--out", default="")
    parser.add_argument("--policy-path", default="")
    args = parser.parse_args()

    out = Path(args.out or f"runs/eval_{args.task}_seed{args.seed}.svg")
    render_eval_rollout(
        task_name=args.task,
        backend=args.backend,
        seed=args.seed,
        budget_seconds=args.budget_seconds,
        out_path=out,
        policy_path=args.policy_path or None,
    )
    print(out)


def render_eval_rollout(
    task_name: str,
    backend: str,
    seed: int,
    budget_seconds: float,
    out_path: Path,
    policy_path: str | None = None,
) -> None:
    task = load_task(task_name)
    budget = TrainBudget(
        seconds=budget_seconds,
        max_iterations=30,
        rollouts_per_iteration=24,
        eval_episodes=1,
    )
    policy = (
        research.load_policy(policy_path)
        if policy_path
        else research.train_policy(task=task, budget=budget, seed=seed, backend=backend)
    )
    env = make_env(task=task, world="eval", seed=seed, backend=backend)
    obs = env.reset()

    ee_points = [_env_ee(env)]
    obj_points = [_env_obj(env)]
    rewards = []
    final_info = {}
    for _ in range(task.horizon):
        result = env.step(research.act(policy, obs))
        obs = result.obs
        rewards.append(result.reward)
        ee_points.append(_env_ee(env))
        obj_points.append(_env_obj(env))
        final_info = result.info
        if result.terminated or result.truncated:
            break

    _write_svg(
        out_path=out_path,
        task_name=f"{backend}/{task_name}",
        ee_points=np.array(ee_points),
        obj_points=np.array(obj_points),
        target=_env_target(env),
        success=bool(final_info.get("success", False)),
        distance=float(final_info.get("distance", 999.0)),
        total_return=float(sum(rewards)),
    )


def _env_ee(env) -> np.ndarray:
    if hasattr(env, "_ee_xy"):
        return env._ee_xy()
    return env.ee.copy()


def _env_obj(env) -> np.ndarray:
    if hasattr(env, "_object_xy"):
        return env._object_xy()
    return env.obj.copy()


def _env_target(env) -> np.ndarray:
    if hasattr(env, "_target_xy"):
        return env._target_xy()
    return env.target.copy()


def _write_svg(
    out_path: Path,
    task_name: str,
    ee_points: np.ndarray,
    obj_points: np.ndarray,
    target: np.ndarray,
    success: bool,
    distance: float,
    total_return: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = 680
    height = 560
    pad = 54
    world_min = -1.25
    world_max = 1.25
    scale = (width - 2 * pad) / (world_max - world_min)

    def xy(point: np.ndarray) -> tuple[float, float]:
        x = pad + (float(point[0]) - world_min) * scale
        y = height - pad - (float(point[1]) - world_min) * scale
        return x, y

    ee_polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, ee_points))
    obj_polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, obj_points))
    target_x, target_y = xy(target)
    start_x, start_y = xy(ee_points[0])
    end_x, end_y = xy(ee_points[-1])
    obj_x, obj_y = xy(obj_points[-1])
    status = "success" if success else "miss"
    status_color = "#2ca02c" if success else "#d62728"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" fill="white" stroke="#d0d0d0"/>',
        f'<text x="24" y="30" font-family="Arial" font-size="18" font-weight="700">Eval rollout: {task_name}</text>',
        f'<text x="24" y="52" font-family="Arial" font-size="12" fill="{status_color}">{status} | distance={distance:.3f} | return={total_return:.3f}</text>',
        f'<polyline points="{ee_polyline}" fill="none" stroke="#1f77b4" stroke-width="3"/>',
        f'<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="5" fill="#8ab6e6"><title>start</title></circle>',
        f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="6" fill="#1f77b4"><title>end effector end</title></circle>',
        f'<circle cx="{target_x:.1f}" cy="{target_y:.1f}" r="9" fill="none" stroke="#2ca02c" stroke-width="3"><title>target</title></circle>',
        f'<line x1="{target_x - 11:.1f}" y1="{target_y:.1f}" x2="{target_x + 11:.1f}" y2="{target_y:.1f}" stroke="#2ca02c" stroke-width="2"/>',
        f'<line x1="{target_x:.1f}" y1="{target_y - 11:.1f}" x2="{target_x:.1f}" y2="{target_y + 11:.1f}" stroke="#2ca02c" stroke-width="2"/>',
    ]
    if task_name == "push":
        parts.extend(
            [
                f'<polyline points="{obj_polyline}" fill="none" stroke="#ff7f0e" stroke-width="2" stroke-dasharray="6 4"/>',
                f'<circle cx="{obj_x:.1f}" cy="{obj_y:.1f}" r="7" fill="#ff7f0e"><title>object end</title></circle>',
            ]
        )
    parts.extend(
        [
            '<text x="520" y="30" font-family="Arial" font-size="12" fill="#1f77b4">blue: end effector</text>',
            '<text x="520" y="48" font-family="Arial" font-size="12" fill="#2ca02c">green: target</text>',
            "</svg>",
        ]
    )
    out_path.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
