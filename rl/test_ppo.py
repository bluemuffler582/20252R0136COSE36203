import os
import numpy as np
import torch
import cv2
import gymnasium as gym

from lib.agent_ppo import PPOAgent
from lib.utils import parse_args_ppo, make_env

from pathlib import Path

# This file lives in rl/, so BASE_DIR is rl/
BASE_DIR = Path(__file__).resolve().parent      # .../PPO-Humanoid/rl
CKPT_DIR = BASE_DIR / "ckpts"                  # .../PPO-Humanoid/rl/ckpts


def make_env_for_video(env_id: str, reward_scale: float, fps: int = 30):
    """
    Create a single environment with render_mode='rgb_array' using the same
    make_env logic as training (so it works for Humanoid-v5, HumanoidTurnRight-v0, etc.).
    """
    env = make_env(env_id, reward_scaling=reward_scale, render=True, fps=fps)
    return env


if __name__ == "__main__":
    args = parse_args_ppo()
    device = torch.device("cuda" if args.cuda else "cpu")

    # ===== basic video settings =====
    fps = 30
    duration_sec = getattr(args, "duration", 5.0)  # default 5s if not provided
    max_steps = int(duration_sec * fps)
    print(f"[INFO] Running {args.env} for {duration_sec:.2f} seconds "
          f"({max_steps} steps at {fps} FPS)")
    # ================================

    # ===== video output path =====
    video_folder = "./videos_eval"
    os.makedirs(video_folder, exist_ok=True)

    dur_str = str(int(duration_sec))
    video_filename = f"ppo_demo_{args.env}_dur{dur_str}s.mp4"
    video_filename = video_filename.replace(":", "-")
    video_path = os.path.join(video_folder, video_filename)
    # =============================

    # ===== 1) Make env for chosen skill =====
    env = make_env_for_video(args.env, reward_scale=args.reward_scale, fps=fps)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ===== 2) Build agent & load checkpoint =====
    agent = PPOAgent(obs_dim, action_dim).to(device)

    # Map env name -> relative checkpoint path under rl/ckpts
    ckpt_map = {
        "Humanoid-v5":             "walk_forward/best.pt",
        "HumanoidWalkBackward-v0": "walk_backward/best.pt",
        "HumanoidTurnRight-v0":    "turn_right/best.pt",
        "HumanoidTurnLeft-v0":     "turn_left/best.pt",
        "HumanoidBalance-v0":      "balance/best.pt",
    }

    if args.env not in ckpt_map:
        raise ValueError(f"No checkpoint mapping registered for env '{args.env}'")

    # Build full path: rl/ckpts/<skill>/best.pt
    checkpoint_path = CKPT_DIR / ckpt_map[args.env]
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["state_dict"])
    else:
        agent.load_state_dict(checkpoint)
    agent.eval()

    # ===== 3) Prepare video writer =====
    obs, _ = env.reset()
    frame = env.render()  # RGB array
    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    episode_reward = 0.0

    # ===== 4) Rollout & record =====
    for step in range(max_steps):
        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        with torch.no_grad():
            obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)

        obs, reward, terminated, truncated, _ = env.step(
            action.squeeze(0).cpu().numpy()
        )
        episode_reward += reward

        if step % 20 == 0:
            print(f"[STEP {step}] reward={reward:.3f}")

        if terminated or truncated:
            print(f"[INFO] Episode ended early at step {step}")
            break

    env.close()
    writer.release()

    print(f"\n[RESULT] Env: {args.env}")
    print(f"[RESULT] Total episode reward (scaled): {episode_reward:.2f}")
    print(f"[RESULT] Video saved at: {video_path}")
