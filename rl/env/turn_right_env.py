import gymnasium as gym
import numpy as np


class HumanoidTurnRightEnv(gym.Wrapper):
    """
    Humanoid-v5 wrapper that learns a TURN-RIGHT-IN-PLACE skill.

    Objectives:
      - Turn RIGHT (clockwise) with the ROOT (whole body yaw, not just waist).
      - Stay as upright and tall as possible.
      - Minimize COM translation (turn around the same spot).
      - Avoid deep squats and bouncing/jumping.
    """

    def __init__(
        self,
        env_id: str = "Humanoid-v5",
        target_yaw_rate: float = -0.7,   # desired yaw rate (rad/s), <0 = turn right
        yaw_k: float = 2.0,              # smoother Gaussian around target_yaw_rate
        yaw_coef: float = 3.0,           # MAIN weight on yaw tracking reward
        move_pen_coef: float = 2.0,      # penalty on COM x,y motion (velocity)
        anchor_coef: float = 1.0,        # NEW: penalty on COM x,y position drift
        twist_coef: float = 0.05,        # penalty on abdomen (waist) yaw velocity
        alive_coef: float = 0.5,         # weight for staying upright (smaller than yaw)
        use_base_reward: bool = False,   # optionally mix in original Humanoid reward
        base_coef: float = 0.1,          # scaling for base_reward if used
        standstill_reset: bool = True,   # zero velocities at reset
        stand_height: float = 1.2,       # desired torso height for "standing"
        squat_coef: float = 3.0,         # penalty for being below stand_height
        jump_coef: float = 0.05,         # penalty on vertical root speed
        render_mode: str | None = None,
    ):
        # Create underlying environment
        env = gym.make(env_id, render_mode=render_mode)
        super().__init__(env)

        # Store shaping parameters
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

    # ------------------------------------------------------------------ #
    # Reset: optionally force a true standstill at episode start
    # ------------------------------------------------------------------ #
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.standstill_reset:
            sim = self.env.unwrapped
            # Zero velocities for a genuine standstill
            sim.data.qvel[:] = 0.0
            # Center x,y so "turn in place" means around (0, 0)
            sim.data.qpos[0] = 0.0  # root x
            sim.data.qpos[1] = 0.0  # root y
            # No sim.forward(): HumanoidEnv doesn't expose the raw MuJoCo sim

        return obs, info

    # ------------------------------------------------------------------ #
    # Step: compute shaped reward for "upright turn in place to the right"
    # ------------------------------------------------------------------ #
    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        sim = self.env.unwrapped
        qpos = np.array(sim.data.qpos, dtype=np.float64)
        qvel = np.array(sim.data.qvel, dtype=np.float64)

        # Root linear & angular velocities
        vx, vy, vz = qvel[0], qvel[1], qvel[2]
        wx, wy, wz = qvel[3], qvel[4], qvel[5]

        # --------------------------------------------------
        # 1) Turn RIGHT with the ROOT (yaw around z-axis)
        # --------------------------------------------------
        yaw_rate = float(wz)  # negative = clockwise (turn right)
        yaw_err = yaw_rate - self.target_yaw_rate

        # (a) Gaussian around target yaw rate
        yaw_gauss = float(np.exp(-self.yaw_k * (yaw_err ** 2)))  # [0, 1]

        # (b) Direction-only term: any negative yaw is good, scaled by target
        #    -yaw_rate > 0 when turning right.
        denom = max(abs(self.target_yaw_rate), 1e-6)
        yaw_dir = np.clip(-yaw_rate / denom, 0.0, 2.0)  # [0, 2] roughly
        # Blend them: stable around target but still rewards “any right turn”.
        yaw_reward = 0.5 * yaw_gauss + 0.5 * (yaw_dir / 2.0)  # keep in ~[0, 1]

        # --------------------------------------------------
        # 2) Penalize COM translation in x,y
        # --------------------------------------------------
        lin_speed_sq = float(vx * vx + vy * vy)
        move_pen = self.move_pen_coef * lin_speed_sq

        # NEW: also penalize position drift so it turns around (0,0)
        root_x = float(qpos[0])
        root_y = float(qpos[1])
        root_xy_dist_sq = root_x * root_x + root_y * root_y
        anchor_pen = self.anchor_coef * root_xy_dist_sq

        # --------------------------------------------------
        # 3) Penalize abdomen-only twisting
        # --------------------------------------------------
        abdomen_z_vel = float(qvel[6])  # waist yaw velocity
        twist_pen = self.twist_coef * abs(abdomen_z_vel)

        # --------------------------------------------------
        # 4) Upright & standing penalties/bonuses
        # --------------------------------------------------
        torso_z = float(qpos[2])

        # Alive bonus: just "not fallen".
        alive_bonus = 1.0 if torso_z > 0.9 else -1.0

        # Squat penalty: if below stand_height, penalize depth.
        squat_depth = max(0.0, self.stand_height - torso_z)
        squat_pen = self.squat_coef * squat_depth

        # Jump penalty: penalize vertical bouncing of the root.
        jump_pen = self.jump_coef * (vz * vz)

        # --------------------------------------------------
        # 5) Combine into final shaped reward
        # --------------------------------------------------
        shaped_reward = 0.0

        # (Optional) Keep a bit of original Humanoid reward
        if self.use_base_reward:
            shaped_reward += self.base_coef * float(base_reward)

        # Turn-right behavior (MAIN term)
        shaped_reward += self.yaw_coef * yaw_reward

        # Regularization terms (movement / twisting / posture)
        shaped_reward -= move_pen
        shaped_reward -= anchor_pen
        shaped_reward -= twist_pen
        shaped_reward -= squat_pen
        shaped_reward -= jump_pen

        # Survival / uprightness
        shaped_reward += self.alive_coef * alive_bonus

        # --------------------------------------------------
        # 6) Diagnostics
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
