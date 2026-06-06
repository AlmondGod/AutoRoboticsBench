from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotbench.config import TaskConfig
from robotbench.envs import StepResult
from robotbench.safety import SafetyTracker


ASSET_DIR = Path(__file__).parent / "assets" / "mujoco"
MODEL_PATH = ASSET_DIR / "planar_arm.xml"


@dataclass(frozen=True)
class MujocoIds:
    shoulder_qpos: int
    elbow_qpos: int
    shoulder_qvel: int
    elbow_qvel: int
    object_qpos: int
    object_qvel: int
    target_body: int
    ee_site: int
    object_body: int
    floor_geom: int
    object_geom: int


class MujocoPlanarRobotEnv:
    """MuJoCo-backed planar manipulation benchmark."""

    obs_dim = 8
    act_dim = 2

    def __init__(self, task: TaskConfig, world: str, seed: int):
        try:
            import mujoco
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MuJoCo backend requires `pip install -e .[mujoco]` or `pip install mujoco`."
            ) from exc

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.task = task
        self.world = world
        self.params = getattr(task, world)
        self.rng = np.random.default_rng(seed)
        self.safety = SafetyTracker(action_limit=task.action_limit)
        self.t = 0
        self.prev_action = np.zeros(2)
        self.action_queue: list[np.ndarray] = []
        self.ids = self._ids()
        self._apply_world_params()

    def reset(self) -> np.ndarray:
        self.mujoco.mj_resetData(self.model, self.data)
        self.t = 0
        self.safety.reset()
        self.prev_action = np.zeros(2)
        latency_steps = int(self.params.get("latency_steps", 0))
        self.action_queue = [np.zeros(2) for _ in range(latency_steps)]

        self.data.qpos[self.ids.shoulder_qpos] = self.rng.uniform(-0.8, 0.8)
        self.data.qpos[self.ids.elbow_qpos] = self.rng.uniform(-0.8, 0.8)
        self.data.qvel[self.ids.shoulder_qvel] = 0.0
        self.data.qvel[self.ids.elbow_qvel] = 0.0

        target = self.rng.uniform([0.25, -0.45], [0.7, 0.45])
        self.model.body_pos[self.ids.target_body, 0:2] = target
        self.model.body_pos[self.ids.target_body, 2] = 0.025

        obj_xy = self.rng.uniform([-0.1, -0.35], [0.35, 0.35])
        q = self.ids.object_qpos
        self.data.qpos[q : q + 7] = np.array([obj_xy[0], obj_xy[1], 0.035, 1.0, 0.0, 0.0, 0.0])
        self.data.qvel[self.ids.object_qvel : self.ids.object_qvel + 6] = 0.0

        self.mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray) -> StepResult:
        self.t += 1
        raw_action = np.asarray(action, dtype=np.float64)
        action = np.clip(raw_action, -self.task.action_limit, self.task.action_limit)
        self.safety.observe(raw_action=raw_action, clipped_action=action)

        if self.action_queue:
            self.action_queue.append(action)
            applied = self.action_queue.pop(0)
        else:
            applied = action

        control_scale = float(self.params.get("control_scale", 1.0))
        self.data.ctrl[:] = control_scale * applied
        frame_skip = int(self.params.get("frame_skip", 5))
        for _ in range(frame_skip):
            self.mujoco.mj_step(self.model, self.data)

        dist = self._distance_to_goal()
        success = dist <= self.task.success_tolerance
        energy = float(np.sum(np.square(applied)))
        jerk = float(np.sum(np.square(applied - self.prev_action)))
        reward = -dist - 0.01 * energy - 0.005 * jerk + (1.0 if success else 0.0)
        self.prev_action = applied

        terminated = bool(success)
        truncated = self.t >= self.task.horizon
        info = {
            "success": success,
            "distance": dist,
            "energy": energy,
            "jerk": jerk,
            **self.safety.snapshot(),
        }
        return StepResult(self._obs(), float(reward), terminated, truncated, info)

    def render_rgb(self, width: int = 720, height: int = 720) -> np.ndarray:
        renderer = self.mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera="top")
        image = renderer.render()
        renderer.close()
        return image

    def _ids(self) -> MujocoIds:
        mujoco = self.mujoco
        shoulder_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder")
        elbow_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")
        object_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        return MujocoIds(
            shoulder_qpos=self.model.jnt_qposadr[shoulder_joint],
            elbow_qpos=self.model.jnt_qposadr[elbow_joint],
            shoulder_qvel=self.model.jnt_dofadr[shoulder_joint],
            elbow_qvel=self.model.jnt_dofadr[elbow_joint],
            object_qpos=self.model.jnt_qposadr[object_joint],
            object_qvel=self.model.jnt_dofadr[object_joint],
            target_body=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target"),
            ee_site=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"),
            object_body=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object"),
            floor_geom=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"),
            object_geom=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"),
        )

    def _apply_world_params(self) -> None:
        friction = float(self.params.get("friction", 1.0))
        object_mass = float(self.params.get("object_mass", 0.12))
        damping = float(self.params.get("damping", 0.12))
        self.model.geom_friction[self.ids.floor_geom, 0] = friction
        self.model.geom_friction[self.ids.object_geom, 0] = friction
        self.model.body_mass[self.ids.object_body] = object_mass
        self.model.dof_damping[self.ids.shoulder_qvel] = damping
        self.model.dof_damping[self.ids.elbow_qvel] = damping

    def _obs(self) -> np.ndarray:
        ee = self._ee_xy()
        obj = self._object_xy() if self.task.name == "push" else np.zeros(2)
        target = self._target_xy()
        subject = obj if self.task.name == "push" else ee
        obs = np.concatenate([ee, obj, target, target - subject]).astype(np.float64)
        noise = float(self.params.get("observation_noise", 0.0))
        if noise:
            obs = obs + self.rng.normal(0.0, noise, size=obs.shape)
        return obs

    def _distance_to_goal(self) -> float:
        subject = self._object_xy() if self.task.name == "push" else self._ee_xy()
        return float(np.linalg.norm(subject - self._target_xy()))

    def _ee_xy(self) -> np.ndarray:
        return self.data.site_xpos[self.ids.ee_site, 0:2].copy()

    def _object_xy(self) -> np.ndarray:
        return self.data.xpos[self.ids.object_body, 0:2].copy()

    def _target_xy(self) -> np.ndarray:
        return self.model.body_pos[self.ids.target_body, 0:2].copy()


def make_mujoco_env(task: TaskConfig, world: str, seed: int) -> MujocoPlanarRobotEnv:
    return MujocoPlanarRobotEnv(task=task, world=world, seed=seed)
