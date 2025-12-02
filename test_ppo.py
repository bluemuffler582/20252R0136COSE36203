import os
import gymnasium as gym
import numpy as np
import torch
import cv2

from lib.agent_ppo import PPOAgent
from lib.utils import parse_args_ppo, make_env


def make_env_for_video(env_id: str, reward_scale: float, fps: int = 30):
    """
    Create a single environment with render_mode='rgb_array' using the same
    make_env logic as training (so it works for Humanoid-v5, HumanoidTurnRight-v0, etc.).
    """
    # render=True -> make_env will set render_mode='rgb_array'
    env = make_env(env_id, reward_scaling=reward_scale, render=True, fps=fps)
    return env


if __name__ == "__main__":
    args = parse_args_ppo()
    device = torch.device("cuda" if args.cuda else "cpu")

    video_folder = "./videos_eval"
    os.makedirs(video_folder, exist_ok=True)
    video_path = os.path.join(video_folder, f"ppo_demo_{args.env}.mp4")

    max_steps = 600
    fps = 30

    # === 1) Make env for chosen skill ===
    env = make_env_for_video(args.env, reward_scale=args.reward_scale, fps=fps)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # === 2) Build agent & load checkpoint ===
    agent = PPOAgent(obs_dim, action_dim).to(device)

    # adjust path if needed
    checkpoint_path = "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["state_dict"])
    else:
        agent.load_state_dict(checkpoint)
    agent.eval()

    # === 3) Prepare video writer ===
    # Get one frame to know size
    obs, _ = env.reset()
    frame = env.render()  # RGB array
    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    episode_reward = 0.0

    for step in range(max_steps):
        # write current frame
        frame = env.render()          # RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        with torch.no_grad():
            obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)

        obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        episode_reward += reward

        if step % 20 == 0:
            print(f"step={step}, reward={reward:.3f}")

        if terminated or truncated:
            print(f"Terminated at step {step}")
            break

    env.close()
    writer.release()

    print(f"\nEnv: {args.env}")
    print(f"Total episode reward (scaled): {episode_reward:.2f}")
    print(f"Video saved at: {video_path}")
