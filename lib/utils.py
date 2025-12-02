import argparse

import cv2
import gymnasium as gym
import numpy as np
import torch

from env.turn_right_env import HumanoidTurnRightEnv
from env.turn_left_env import HumanoidTurnLeftEnv
from env.walk_backward_env import HumanoidWalkBackwardEnv



def parse_args_ppo() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True if torch.cuda.is_available() else False, action="store_true",
                        help="Use CUDA")
    parser.add_argument("--env", default="Humanoid-v5", help="Environment to use")
    parser.add_argument("--n-envs", type=int, default=32, help="Number of environments")
    parser.add_argument("--n-epochs", type=int, default=1000, help="Number of epochs to run")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps per epoch per environment")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--train-iters", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Lambda for GAE")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--target-kl", type=float, default=0.01, help="Target KL divergence")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--reward-scale", type=float, default=0.01, help="Reward scaling")
    parser.add_argument("--render-epoch", type=int, default=20, help="Render every n-th epoch")
    return parser.parse_args()


def log_video(env, agent, device, video_path, fps=30):
    """
    Log a video of one episode of the agent playing in the environment.
    :param env: a test environment which supports video recording and doesn't conflict with the other environments.
    :param agent: the agent to record.
    :param device: the device to run the agent on.
    :param video_path: the path to save the video.
    :param fps: the frames per second of the video.
    """
    agent.eval()
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        # Render the frame
        frames.append(env.render())
        # Sample an action
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.tensor(np.array([obs], dtype=np.float32), device=device))
        # Step the environment
        obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated
    # Save the video
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

def make_env(env_id, reward_scaling=0.01, render=False, fps=30):
    """
    Make an environment with the given id.
    """

    # Special case: our custom "turn right in place" env
    if env_id == "HumanoidTurnRight-v0":
        common_kwargs = dict(
            env_id="Humanoid-v5",
            target_yaw_rate=-0.7,   # slight but clear right turn
            yaw_k=2.0,              # smoother Gaussian
            yaw_coef=3.0,           # make yaw dominant
            move_pen_coef=2.0,      # strong penalty on xy velocity
            anchor_coef=1.0,        # NEW: keep COM near origin
            twist_coef=0.05,        # allow some waist motion but not too much
            standstill_reset=True,
            stand_height=1.25,
            squat_coef=5.0,
            jump_coef=0.05,
        )
    
        if render:
            env = HumanoidTurnRightEnv(
                render_mode="rgb_array",
                **common_kwargs,
            )
            env.metadata["render_fps"] = fps
        else:
            env = HumanoidTurnRightEnv(
                render_mode=None,
                **common_kwargs,
            )
    
        # still apply your global reward scaling
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
        return env

    if env_id == "HumanoidTurnLeft-v0":
        common_kwargs = dict(
            env_id="Humanoid-v5",
            target_yaw_rate=+0.7,   # mirrored direction (turn LEFT)
            yaw_k=2.0,
            yaw_coef=3.0,
            move_pen_coef=2.0,
            anchor_coef=1.0,
            twist_coef=0.05,
            standstill_reset=True,
            stand_height=1.25,
            squat_coef=5.0,
            jump_coef=0.05,
        )
    
        if render:
            env = HumanoidTurnLeftEnv(
                render_mode="rgb_array",
                **common_kwargs,
            )
            env.metadata["render_fps"] = fps
        else:
            env = HumanoidTurnLeftEnv(
                render_mode=None,
                **common_kwargs,
            )
    
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
        return env

    if env_id == "HumanoidWalkBackward-v0":
        common_kwargs = dict(
            env_id="Humanoid-v5",
            target_speed=1.0,
            w_speed=1.0,
            w_alive=5.0,
            ctrl_cost_weight=0.5e-3,
            impact_cost_weight=5e-7,
            impact_cost_cap=0.1,
            height_target=1.4,
            height_tolerance=0.4,
            w_upright=0.5,
            min_height_for_alive=0.8,
        )

        if render:
            env = HumanoidWalkBackwardEnv(
                render_mode="rgb_array",
                **common_kwargs,
            )
            env.metadata["render_fps"] = fps
        else:
            env = HumanoidWalkBackwardEnv(
                render_mode=None,
                **common_kwargs,
            )

        # Apply global reward scaling (same as your TurnLeft/Right)
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
        return env


    # Default: original behaviour (e.g. Humanoid-v5 walking forward)
    if render:
        env = gym.make(env_id, render_mode='rgb_array')
        env.metadata['render_fps'] = fps
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    else:
        env = gym.make(env_id)
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    return env




''' normal walk forward
def make_env(env_id, reward_scaling=0.01, render=False, fps=30):
    """
    Make an environment with the given id.
    :param env_id: the id of the environment.
    :param reward_scaling: the scaling factor for the rewards.
    :param render: whether to render the environment.
    :param fps: the frames per second if rendering.
    :return: the environment.
    """
    if render:
        env = gym.make(env_id, render_mode='rgb_array')
        env.metadata['render_fps'] = fps
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    else:
        env = gym.make(env_id)
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    return env
'''