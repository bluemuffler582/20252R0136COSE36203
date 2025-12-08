# env/balance_env.py

import gymnasium as gym
import numpy as np


class HumanoidBalanceEnv(gym.Wrapper):
    """
    Humanoid-v5 wrapper that learns a pure BALANCE skill.

    Goal:
      - Stay upright and not fall.
      - Keep torso height near a target.
      - Avoid excessive COM velocity and huge torques.

    Notes
    -----
    - qpos[2] ≈ torso/root height
    - qvel[0:3] = root linear velocity (vx, vy, vz)
    """

    def __init__(
        self,
        env_id: str = "Humanoid-v5",
        render_mode=None,
        target_height: float = 1.25,
        max_steps: int = 1000,
    ):
        # Create the underlying Mujoco env
        env = gym.make(env_id, render_mode=render_mode)
        super().__init__(env)

        self.target_height = target_height
        self.max_steps = max_steps
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def _compute_reward(self, action):
        # Access mujoco data from the unwrapped env
        data = self.env.unwrapped.data
        qpos = data.qpos
        qvel = data.qvel

        # --- Height / upright proxy ---
        torso_z = float(qpos[2])

        # Height close to target → reward ≈ 1, farther → decays
        height_err = abs(torso_z - self.target_height)
        height_reward = np.exp(-height_err)

        # Normalized "upright" term based only on height (0–1)
        # 0 if below 0.8, 1 around target_height.
        upright_reward = np.clip(
            (torso_z - 0.8) / (self.target_height - 0.8 + 1e-6),
            0.0,
            1.0,
        )

        # --- COM / root velocity penalty ---
        root_lin_vel = qvel[0:3]
        com_speed = float(np.linalg.norm(root_lin_vel))
        vel_penalty = com_speed

        # --- Action penalty (smoothness) ---
        action_penalty = float(np.square(action).sum())

        # --- Combine ---
        reward = (
            1.0  # alive bonus
            + 2.0 * upright_reward
            + 1.0 * height_reward
            - 0.1 * vel_penalty
            - 0.001 * action_penalty
        )

        return reward, torso_z

    def step(self, action):
        self._step_count += 1

        # Step base env, ignore its reward and recompute ours
        obs, _, terminated, truncated, info = self.env.step(action)

        reward, torso_z = self._compute_reward(action)

        # Terminate if we "fall"
        fallen = torso_z < 0.8
        terminated = bool(terminated or fallen)

        # Truncate on max steps
        truncated = bool(truncated or (self._step_count >= self.max_steps))

        # Debug info
        info = dict(info)
        info["fallen"] = fallen
        info["torso_z"] = torso_z

        return obs, reward, terminated, truncated, info
