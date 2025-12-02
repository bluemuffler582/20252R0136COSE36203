# env/walk_backward_env.py

import gymnasium as gym
import numpy as np


class HumanoidWalkBackwardEnv(gym.Wrapper):
    """
    Wrapper around Humanoid-v5 that encourages WALKING BACKWARD.

    Idea:
      - Keep the same *style* of reward terms as forward walking:
        speed + alive - control cost - impact cost (+ upright).
      - BUT use backward speed (negative vx) instead of forward vx.
    """

    def __init__(
        self,
        env_id: str = "Humanoid-v5",
        target_speed: float = 1.0,       # desired backward speed in m/s
        w_speed: float = 1.0,            # scaling for speed reward
        w_alive: float = 5.0,            # alive bonus (similar to classic Humanoid)
        ctrl_cost_weight: float = 0.5e-3,
        impact_cost_weight: float = 5e-7,
        impact_cost_cap: float = 0.1,
        height_target: float = 1.4,
        height_tolerance: float = 0.4,
        w_upright: float = 0.5,
        min_height_for_alive: float = 0.8,
        render_mode: str | None = None,
    ):
        env = gym.make(env_id, render_mode=render_mode)
        super().__init__(env)

        self.target_speed = target_speed
        self.w_speed = w_speed
        self.w_alive = w_alive
        self.ctrl_cost_weight = ctrl_cost_weight
        self.impact_cost_weight = impact_cost_weight
        self.impact_cost_cap = impact_cost_cap

        self.height_target = height_target
        self.height_tolerance = height_tolerance
        self.w_upright = w_upright
        self.min_height_for_alive = min_height_for_alive

        # indices for convenience (Humanoid: qpos[0]=x, [1]=y, [2]=z)
        self._height_idx = 2

    # Gymnasium API

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # Step underlying env first (so mujoco state is updated)
        obs, _, terminated, truncated, info = self.env.step(action)

        data = self.env.unwrapped.data

        # ------------------------------------------------------------------
        # 1) Root velocity (vx)  -> we want NEGATIVE vx (backward)
        # ------------------------------------------------------------------
        vx = float(data.qvel[0])  # root linear velocity in x
        backward_speed = max(-vx, 0.0)  # only reward if actually going backward

        # Normalize by target speed and clip to [0, 1]
        speed_term = np.clip(backward_speed / self.target_speed, 0.0, 1.0)
        r_speed = self.w_speed * speed_term

        # ------------------------------------------------------------------
        # 2) Alive bonus & fall condition (height-based)
        # ------------------------------------------------------------------
        height = float(data.qpos[self._height_idx])
        alive = height > self.min_height_for_alive
        r_alive = self.w_alive if alive else 0.0

        # If height too low, treat as fall
        if not alive:
            terminated = True

        # ------------------------------------------------------------------
        # 3) Upright / height term (keep torso near some target height)
        # ------------------------------------------------------------------
        height_err = abs(height - self.height_target)
        if height_err < self.height_tolerance:
            upright_factor = 1.0 - (height_err / self.height_tolerance)
        else:
            upright_factor = 0.0
        r_upright = self.w_upright * upright_factor

        # ------------------------------------------------------------------
        # 4) Control cost (penalize large actions)
        #    ~ 0.5e-3 * ||a||^2 like classic Humanoid
        # ------------------------------------------------------------------
        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(action)))
        r_ctrl = -ctrl_cost

        # ------------------------------------------------------------------
        # 5) Impact cost (penalize large external contact forces)
        # ------------------------------------------------------------------
        # data.cfrc_ext: (nbody, 6) external forces, classic gym uses this
        if hasattr(data, "cfrc_ext"):
            impact_cost = self.impact_cost_weight * float(
                np.sum(np.square(data.cfrc_ext))
            )
            impact_cost = min(impact_cost, self.impact_cost_cap)
        else:
            impact_cost = 0.0
        r_impact = -impact_cost

        # ------------------------------------------------------------------
        # 6) Total custom reward
        # ------------------------------------------------------------------
        reward = r_speed + r_alive + r_upright + r_ctrl + r_impact

        # attach diagnostics
        info = dict(info)
        info.update(
            vx=vx,
            backward_speed=backward_speed,
            r_speed=r_speed,
            r_alive=r_alive,
            r_upright=r_upright,
            r_ctrl=r_ctrl,
            r_impact=r_impact,
            height=height,
        )

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
