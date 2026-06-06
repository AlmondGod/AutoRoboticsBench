from __future__ import annotations

from pathlib import Path

import numpy as np

from robotbench.config import TaskConfig
from robotbench.envs import StepResult
from robotbench.safety import SafetyTracker


MENAGERIE_DIR = Path("third_party/mujoco_menagerie")
ALOHA_DIR = MENAGERIE_DIR / "aloha"
GENERATED_SCENE = ALOHA_DIR / "robot_autoresearch_scene.xml"
GENERATED_MOBILE_MOCK_SCENE = ALOHA_DIR / "robot_autoresearch_mobile_mock_scene.xml"

ACTUATOR_NAMES = [
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
    "left/gripper",
    "right/waist",
    "right/shoulder",
    "right/elbow",
    "right/forearm_roll",
    "right/wrist_angle",
    "right/wrist_rotate",
    "right/gripper",
]


class AlohaBimanualEnv:
    """Menagerie ALOHA bimanual environment with a 14-DOF action surface."""

    obs_dim = 8
    act_dim = 14
    ctrl_dim = 14

    def __init__(self, task: TaskConfig, world: str, seed: int):
        try:
            import mujoco
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ALOHA backend requires `pip install -e .[mujoco]` or `pip install mujoco`."
            ) from exc

        if not (ALOHA_DIR / "aloha.xml").exists():
            raise FileNotFoundError(
                "Missing Menagerie ALOHA assets. Run: "
                "python scripts/fetch_menagerie.py --model aloha"
            )

        _ensure_generated_scene()
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(self._scene_path()))
        self.data = mujoco.MjData(self.model)
        self.task = task
        self.world = world
        self.params = getattr(task, world)
        self.rng = np.random.default_rng(seed)
        self.safety = SafetyTracker(action_limit=task.action_limit)
        self.t = 0
        self.prev_action = np.zeros(self.act_dim)
        self.nu = self.model.nu
        if self.nu != self.ctrl_dim:
            raise ValueError(f"expected 14 ALOHA actuators, got {self.nu}")
        self.ctrlrange = self.model.actuator_ctrlrange.copy()
        self.neutral_ctrl = np.zeros(self.ctrl_dim)
        key_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, "neutral_pose")
        if key_id >= 0:
            self.neutral_ctrl = self.model.key_ctrl[key_id].copy()
        self.left_site = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, "left/gripper")
        self.right_site = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, "right/gripper")
        self.target_body = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "autoresearch_target")

    def _scene_path(self) -> Path:
        return GENERATED_SCENE

    def reset(self) -> np.ndarray:
        self.mujoco.mj_resetData(self.model, self.data)
        key_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, "neutral_pose")
        if key_id >= 0:
            self.mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.t = 0
        self.safety.reset()
        self.prev_action = np.zeros(self.act_dim)

        if self.world == "eval":
            target = self.rng.uniform([-0.18, -0.03, 0.18], [0.18, 0.18, 0.38])
        else:
            target = self.rng.uniform([-0.16, -0.01, 0.2], [0.16, 0.14, 0.36])
        self.model.body_pos[self.target_body] = target
        self.mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray) -> StepResult:
        self.t += 1
        raw_action = np.asarray(action, dtype=np.float64).reshape(-1)
        if raw_action.size < self.act_dim:
            raw_action = np.pad(raw_action, (0, self.act_dim - raw_action.size))
        raw_action = raw_action[: self.act_dim]
        action = np.clip(raw_action, -self.task.action_limit, self.task.action_limit)
        self.safety.observe(raw_action=raw_action, clipped_action=action)
        self._apply_base_action(action)
        ctrl_action = self._ctrl_action(action)

        lo = self.ctrlrange[:, 0]
        hi = self.ctrlrange[:, 1]
        span = hi - lo
        scale = 0.18 if self.world == "train" else 0.14
        ctrl = np.clip(self.neutral_ctrl + scale * span * ctrl_action, lo, hi)
        self.data.ctrl[:] = ctrl
        for _ in range(int(self.params.get("frame_skip", 8))):
            self.mujoco.mj_step(self.model, self.data)

        dist = self._distance_to_goal()
        success = dist <= self.task.success_tolerance
        energy = float(np.sum(np.square(action)))
        jerk = float(np.sum(np.square(action - self.prev_action)))
        reward = -dist - 0.01 * energy - 0.002 * jerk + (1.0 if success else 0.0)
        self.prev_action = action
        truncated = self.t >= self.task.horizon
        info = {
            "success": success,
            "distance": dist,
            "energy": energy,
            "jerk": jerk,
            **self.safety.snapshot(),
        }
        return StepResult(self._obs(), float(reward), bool(success), truncated, info)

    def render_rgb(self, width: int = 1280, height: int = 720) -> np.ndarray:
        renderer = self.mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera="overhead_cam")
        image = renderer.render()
        renderer.close()
        return image

    def _obs(self) -> np.ndarray:
        left = self._left_xy()
        right = self._right_xy()
        target = self._target_xy()
        obs = np.concatenate([left, right, target, target - left]).astype(np.float64)
        noise = float(self.params.get("observation_noise", 0.0))
        if noise:
            obs = obs + self.rng.normal(0.0, noise, size=obs.shape)
        return obs

    def _apply_base_action(self, action: np.ndarray) -> None:
        del action

    def _ctrl_action(self, action: np.ndarray) -> np.ndarray:
        return action[: self.ctrl_dim]

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.data.site_xpos[self.left_site] - self.model.body_pos[self.target_body]))

    def _ee_xy(self) -> np.ndarray:
        return self._left_xy()

    def _object_xy(self) -> np.ndarray:
        return self._right_xy()

    def _target_xy(self) -> np.ndarray:
        return self.model.body_pos[self.target_body, 0:2].copy()

    def _left_xy(self) -> np.ndarray:
        return self.data.site_xpos[self.left_site, 0:2].copy()

    def _right_xy(self) -> np.ndarray:
        return self.data.site_xpos[self.right_site, 0:2].copy()


class MobileAlohaMockEnv(AlohaBimanualEnv):
    """Mock mobile ALOHA setting: Menagerie ALOHA arms on a kinematic base."""

    act_dim = 17

    def __init__(self, task: TaskConfig, world: str, seed: int):
        super().__init__(task=task, world=world, seed=seed)
        obj = self.mujoco.mjtObj
        self.left_base = self.mujoco.mj_name2id(self.model, obj.mjOBJ_BODY, "left/base_link")
        self.right_base = self.mujoco.mj_name2id(self.model, obj.mjOBJ_BODY, "right/base_link")
        self.mobile_base = self.mujoco.mj_name2id(self.model, obj.mjOBJ_BODY, "autoresearch_mobile_base")
        self.left_base_initial = self.model.body_pos[self.left_base].copy()
        self.right_base_initial = self.model.body_pos[self.right_base].copy()
        self.base_xy = np.zeros(2)

    def _scene_path(self) -> Path:
        _ensure_generated_mobile_mock_scene()
        return GENERATED_MOBILE_MOCK_SCENE

    def reset(self) -> np.ndarray:
        obs = super().reset()
        self.base_xy = self.rng.uniform([-0.08, -0.05], [0.08, 0.05])
        self._sync_mobile_base()
        self.mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def _apply_base_action(self, action: np.ndarray) -> None:
        base_action = action[:3]
        slip = 0.85 if self.world == "eval" else 1.0
        self.base_xy += slip * 0.015 * base_action[:2]
        self.base_xy = np.clip(self.base_xy, [-0.35, -0.22], [0.35, 0.22])
        self._sync_mobile_base()

    def _ctrl_action(self, action: np.ndarray) -> np.ndarray:
        return action[3 : 3 + self.ctrl_dim]

    def _sync_mobile_base(self) -> None:
        offset = np.array([self.base_xy[0], self.base_xy[1], 0.0])
        self.model.body_pos[self.left_base] = self.left_base_initial + offset
        self.model.body_pos[self.right_base] = self.right_base_initial + offset
        self.model.body_pos[self.mobile_base, 0:2] = self.base_xy

    def _obs(self) -> np.ndarray:
        base = self.base_xy
        left = self._left_xy()
        target = self._target_xy()
        obs = np.concatenate([base, left, target, target - left]).astype(np.float64)
        noise = float(self.params.get("observation_noise", 0.0))
        if noise:
            obs = obs + self.rng.normal(0.0, noise, size=obs.shape)
        return obs


def _ensure_generated_scene() -> None:
    if GENERATED_SCENE.exists():
        return
    scene = (ALOHA_DIR / "scene.xml").read_text()
    scene = scene.replace(
        "<global azimuth=\"90\" elevation=\"-20\"/>",
        "<global azimuth=\"90\" elevation=\"-20\" offwidth=\"1280\" offheight=\"720\"/>",
    )
    target = """
    <body name="autoresearch_target" pos="0 0.1 0.25">
      <geom name="autoresearch_target_geom" type="sphere" size="0.035" rgba="0.1 0.8 0.2 0.65" contype="0" conaffinity="0"/>
    </body>
"""
    scene = scene.replace("  </worldbody>", target + "  </worldbody>")
    GENERATED_SCENE.write_text(scene)


def _ensure_generated_mobile_mock_scene() -> None:
    if GENERATED_MOBILE_MOCK_SCENE.exists():
        return
    _ensure_generated_scene()
    scene = GENERATED_SCENE.read_text()
    mobile_base = """
    <body name="autoresearch_mobile_base" pos="0 0 -0.02">
      <geom name="autoresearch_mobile_base_geom" type="box" size="0.58 0.32 0.035" rgba="0.08 0.1 0.12 0.55" contype="0" conaffinity="0"/>
      <geom name="autoresearch_mobile_base_wheel_l" type="cylinder" size="0.045 0.025" pos="-0.42 0 0" euler="1.5708 0 0" rgba="0.02 0.02 0.02 1" contype="0" conaffinity="0"/>
      <geom name="autoresearch_mobile_base_wheel_r" type="cylinder" size="0.045 0.025" pos="0.42 0 0" euler="1.5708 0 0" rgba="0.02 0.02 0.02 1" contype="0" conaffinity="0"/>
    </body>
"""
    scene = scene.replace("  </worldbody>", mobile_base + "  </worldbody>")
    GENERATED_MOBILE_MOCK_SCENE.write_text(scene)


def make_aloha_env(task: TaskConfig, world: str, seed: int) -> AlohaBimanualEnv:
    return AlohaBimanualEnv(task=task, world=world, seed=seed)


def make_mobile_aloha_mock_env(task: TaskConfig, world: str, seed: int) -> MobileAlohaMockEnv:
    return MobileAlohaMockEnv(task=task, world=world, seed=seed)
