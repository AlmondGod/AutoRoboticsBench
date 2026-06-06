from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
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
    parser.add_argument("--out", default="runs/eval_rollout.mp4")
    parser.add_argument("--policy-path", default="")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    out_path = Path(args.out)
    render_video(
        task_name=args.task,
        backend=args.backend,
        seed=args.seed,
        budget_seconds=args.budget_seconds,
        out_path=out_path,
        fps=args.fps,
        policy_path=args.policy_path or None,
    )
    print(out_path)


def render_video(
    task_name: str,
    backend: str,
    seed: int,
    budget_seconds: float,
    out_path: Path,
    fps: int,
    policy_path: str | None = None,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to render MP4 video")

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

    frames = []
    trail = [_env_ee(env)]
    obj_trail = [_env_obj(env)]
    final_info = {}
    for step_idx in range(task.horizon):
        result = env.step(research.act(policy, obs))
        obs = result.obs
        trail.append(_env_ee(env))
        obj_trail.append(_env_obj(env))
        final_info = result.info
        if backend in {"mujoco", "aloha", "mobile_aloha_mock"}:
            width, height = (1280, 720) if backend in {"aloha", "mobile_aloha_mock"} else (720, 720)
            frame = env.render_rgb(width=width, height=height)
            _status_bar(
                frame,
                success=bool(final_info.get("success", False)),
            )
            frames.append(frame)
        else:
            frames.append(
                _draw_frame(
                    task_name=task_name,
                    step_idx=step_idx,
                    ee=np.array(trail),
                    obj=np.array(obj_trail),
                    target=_env_target(env),
                    success=bool(final_info.get("success", False)),
                    distance=float(final_info.get("distance", 999.0)),
                )
            )
        if result.terminated or result.truncated:
            break

    # Hold the final state briefly so success/failure is easy to see.
    if frames:
        frames.extend([frames[-1].copy() for _ in range(fps)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for idx, frame in enumerate(frames):
            _write_ppm(tmp_path / f"frame_{idx:04d}.ppm", frame)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_path / "frame_%04d.ppm"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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


def _status_bar(image: np.ndarray, success: bool) -> None:
    color = (44, 160, 44) if success else (214, 39, 40)
    image[0:14, :] = color


def _draw_frame(
    task_name: str,
    step_idx: int,
    ee: np.ndarray,
    obj: np.ndarray,
    target: np.ndarray,
    success: bool,
    distance: float,
) -> np.ndarray:
    width, height = 720, 720
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    world_min, world_max = -1.25, 1.25
    pad = 70
    scale = (width - 2 * pad) / (world_max - world_min)

    def xy(point: np.ndarray) -> tuple[int, int]:
        x = pad + (float(point[0]) - world_min) * scale
        y = height - pad - (float(point[1]) - world_min) * scale
        return int(round(x)), int(round(y))

    _rect(image, pad, pad, width - pad, height - pad, (255, 255, 255))
    _grid(image, pad, width - pad, pad, height - pad, 5)

    base = xy(np.array([0.0, 0.0]))
    target_xy = xy(target)
    current = xy(ee[-1])

    for a, b in zip(ee[:-1], ee[1:], strict=False):
        _line(image, xy(a), xy(b), (100, 155, 210), 2)

    _line(image, base, current, (45, 79, 110), 5)
    _circle(image, base, 9, (40, 40, 40))
    _circle(image, current, 11, (31, 119, 180))
    _target(image, target_xy, 15, (44, 160, 44))

    if task_name == "push":
        for a, b in zip(obj[:-1], obj[1:], strict=False):
            _line(image, xy(a), xy(b), (255, 170, 80), 2)
        _circle(image, xy(obj[-1]), 10, (255, 127, 14))

    status_color = (44, 160, 44) if success else (214, 39, 40)
    _text_bar(
        image,
        f"eval rollout: {task_name} | step {step_idx:03d} | distance {distance:.3f}",
        status_color,
    )
    return image


def _write_ppm(path: Path, image: np.ndarray) -> None:
    h, w, _ = image.shape
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(image.tobytes())


def _grid(image: np.ndarray, x0: int, x1: int, y0: int, y1: int, count: int) -> None:
    for idx in range(count + 1):
        x = x0 + (x1 - x0) * idx // count
        y = y0 + (y1 - y0) * idx // count
        _line(image, (x, y0), (x, y1), (230, 230, 230), 1)
        _line(image, (x0, y), (x1, y), (230, 230, 230), 1)


def _rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    image[y0:y1, x0:x1] = color


def _target(image: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
    _circle_outline(image, center, radius, color, 3)
    _line(image, (center[0] - radius, center[1]), (center[0] + radius, center[1]), color, 2)
    _line(image, (center[0], center[1] - radius), (center[0], center[1] + radius), color, 2)


def _circle(image: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
    cx, cy = center
    x0 = max(0, cx - radius)
    x1 = min(image.shape[1], cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(image.shape[0], cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    y, x = np.ogrid[y0 - cy : y1 - cy, x0 - cx : x1 - cx]
    mask = x * x + y * y <= radius * radius
    image[y0:y1, x0:x1][mask] = color


def _circle_outline(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    cx, cy = center
    outer = radius
    x0 = max(0, cx - outer)
    x1 = min(image.shape[1], cx + outer + 1)
    y0 = max(0, cy - outer)
    y1 = min(image.shape[0], cy + outer + 1)
    if x0 >= x1 or y0 >= y1:
        return
    y, x = np.ogrid[y0 - cy : y1 - cy, x0 - cx : x1 - cx]
    dist = x * x + y * y
    mask = (radius - width) ** 2 <= dist
    mask &= dist <= radius * radius
    image[y0:y1, x0:x1][mask] = color


def _line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for idx in range(steps + 1):
        x = int(round(x0 + (x1 - x0) * idx / steps))
        y = int(round(y0 + (y1 - y0) * idx / steps))
        _circle(image, (x, y), max(1, width // 2), color)


def _text_bar(image: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    # Minimal status strip without a font dependency.
    image[0:48, :] = (245, 245, 245)
    image[42:48, :] = color
    # Encode progress text through terminal/logs; the visual strip carries status.
    del text


if __name__ == "__main__":
    main()
