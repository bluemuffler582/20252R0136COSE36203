import gymnasium as gym
import numpy as np


class HumanoidTurnLeftEnv(gym.Wrapper):
    """
    Humanoid-v5 wrapper that learns a TURN-LEFT-IN-PLACE skill.

    Objectives:
      - Turn LEFT (counterclockwise) with the ROOT (whole body yaw).
      - Stay upright and as tall as possible.
      - Minimize COM translation (turn in place).
      - Avoid squatting deeply or bouncing.
    """

    def __init__(
        self,
        env_id: str = "Humanoid-v5",
        target_yaw_rate: float = +0.7,   # positive = turn left
        yaw_k: float = 2.0,
        yaw_coef: float = 3.0,
        move_pen_coef: float = 2.0,
        anchor_coef: float = 1.0,
        twist_coef: float = 0.05,
        alive_coef: float = 0.5,
        use_base_reward: bool = False,
        base_coef: float = 0.1,
        standstill_reset: bool = True,
        stand_height: float = 1.2,
        squat_coef: float = 3.0,
        jump_coef: float = 0.05,
        render_mode: str | None = None,
    ):
        env = gym.make(env_id, render_mode=render_mode)
        super().__init__(env)

        self.target_yaw_rate = float(target_yaw_rate)
        self.yaw_k = float(yaw_k)
        self.yaw_coef = float(yaw_coef)
        self.move_pen_coef = float(move_pen_coef)
        self.anchor_coef = float(anchor_coef)
        self.twist_coef = float(twist_coef)
        self.alive_coef = float(alive_coef)

        self.use_base_reward = bool(use_base_reward)
        self.base_coef = float(base_coef)
        self.standstill_reset = bool(standstill_reset)

        self.stand_height = float(stand_height)
        self.squat_coef = float(squat_coef)
        self.jump_coef = float(jump_coef)

    # ------------------------------------------------------------ #
    # Reset: zero velocities for cleaner standstill turn-in-place
    # ------------------------------------------------------------ #
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.standstill_reset:
            sim = self.env.unwrapped
            sim.data.qvel[:] = 0.0
            sim.data.qpos[0] = 0.0
            sim.data.qpos[1] = 0.0

        return obs, info

    # ------------------------------------------------------------ #
    # Step: reward for upright turn-left (counterclockwise)
    # ------------------------------------------------------------ #
    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        sim = self.env.unwrapped
        qpos = np.array(sim.data.qpos, dtype=np.float64)
        qvel = np.array(sim.data.qvel, dtype=np.float64)

        # Root velocities
        vx, vy, vz = qvel[0], qvel[1], qvel[2]
        wx, wy, wz = qvel[3], qvel[4], qvel[5]

        # --------------------------------------------------
        # 1) TURN LEFT (positive yaw around z-axis)
        # --------------------------------------------------
        yaw_rate = float(wz)  # +wz = CCW turn-left
        yaw_err = yaw_rate - self.target_yaw_rate

        # Gaussian around target yaw rate
        yaw_gauss = float(np.exp(-self.yaw_k * (yaw_err ** 2)))

        # “Any positive yaw is good” directional term
        denom = max(abs(self.target_yaw_rate), 1e-6)
        yaw_dir = np.clip(yaw_rate / denom, 0.0, 2.0)  # positive yaw only

        yaw_reward = 0.5 * yaw_gauss + 0.5 * (yaw_dir / 2.0)

        # --------------------------------------------------
        # 2) Penalize COM translation
        # --------------------------------------------------
        lin_speed_sq = float(vx * vx + vy * vy)
        move_pen = self.move_pen_coef * lin_speed_sq

        # Also penalize drift from origin
        root_x = float(qpos[0])
        root_y = float(qpos[1])
        root_xy_dist_sq = root_x * root_x + root_y * root_y
        anchor_pen = self.anchor_coef * root_xy_dist_sq

        # --------------------------------------------------
        # 3) Penalize twisting only the abdomen
        # --------------------------------------------------
        abdomen_z_vel = float(qvel[6])
        twist_pen = self.twist_coef * abs(abdomen_z_vel)

        # --------------------------------------------------
        # 4) Uprightness & stable standing
        # --------------------------------------------------
        torso_z = float(qpos[2])
        alive_bonus = 1.0 if torso_z > 0.9 else -1.0

        squat_depth = max(0.0, self.stand_height - torso_z)
        squat_pen = self.squat_coef * squat_depth

        jump_pen = self.jump_coef * (vz * vz)

        # --------------------------------------------------
        # 5) Combine into final reward
        # --------------------------------------------------
        shaped_reward = 0.0

        if self.use_base_reward:
            shaped_reward += self.base_coef * float(base_reward)

        shaped_reward += self.yaw_coef * yaw_reward

        shaped_reward -= move_pen
        shaped_reward -= anchor_pen
        shaped_reward -= twist_pen
        shaped_reward -= squat_pen
        shaped_reward -= jump_pen

        shaped_reward += self.alive_coef * alive_bonus

        # --------------------------------------------------
        # 6) Logging
        # --------------------------------------------------
        info = dict(info)
        info["yaw_rate"] = yaw_rate
        info["yaw_err"] = yaw_err
        info["yaw_gauss"] = yaw_gauss
        info["yaw_dir"] = float(yaw_dir)
        info["yaw_reward"] = yaw_reward
        info["lin_speed_sq"] = lin_speed_sq
        info["root_xy_dist_sq"] = root_xy_dist_sq
        info["abdomen_z_vel"] = abdomen_z_vel
        info["torso_z"] = torso_z
        info["alive_bonus"] = alive_bonus
        info["squat_pen"] = squat_pen
        info["jump_pen"] = jump_pen
        info["anchor_pen"] = anchor_pen
        info["base_reward"] = float(base_reward)
        info["shaped_reward"] = shaped_reward

        return obs, shaped_reward, terminated, truncated, info
