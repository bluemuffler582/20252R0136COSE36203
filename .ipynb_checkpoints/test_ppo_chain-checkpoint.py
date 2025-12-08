import os
import cv2
import torch
import numpy as np
import gymnasium as gym

from lib.agent_ppo import PPOAgent
from lib.utils import make_env, parse_args_ppo


def make_env_for_video(env_id: str, reward_scale: float, fps: int = 30):
    """
    Create a single environment with render_mode='rgb_array' using the same
    make_env logic as training.
    """
    return make_env(env_id, reward_scaling=reward_scale, render=True, fps=fps)


def run_segment(env, agent, writer, device, fps, duration_sec, obs, label="segment"):
    """
    Run one skill for `duration_sec` seconds *without resetting* the env.
    Uses `obs` as the starting observation and returns:
        new_obs, total_reward, done_flag
    """
    max_steps = int(duration_sec * fps)
    total_reward = 0.0
    done = False

    for step in range(max_steps):
        # Render + write frame
        frame = env.render()  # RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        # Policy action
        with torch.no_grad():
            obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)

        obs, reward, terminated, truncated, _ = env.step(
            action.squeeze(0).cpu().numpy()
        )
        total_reward += reward

        if step % 20 == 0:
            print(f"[{label}] step={step}, reward={reward:.3f}")

        if terminated or truncated:
            print(f"[{label}] episode ended early at step {step}")
            done = True
            break

    return obs, total_reward, done


def run_transition(env, agent_from, agent_to, writer, device,
                   fps, duration_sec, obs, label="transition"):
    """
    Smoothly transition from agent_from to agent_to over `duration_sec`.
    The action is a linear blend of the two agents' actions:
        a = (1 - alpha) * a_from + alpha * a_to
    alpha goes from 0 -> 1 over the transition steps.
    """
    max_steps = int(duration_sec * fps)
    total_reward = 0.0
    done = False

    if max_steps <= 0:
        return obs, total_reward, done

    for step in range(max_steps):
        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        with torch.no_grad():
            obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
            a_from, _, _, _ = agent_from.get_action_and_value(obs_tensor)
            a_to,   _, _, _ = agent_to.get_action_and_value(obs_tensor)

        # alpha from 0 -> 1
        alpha = float(step + 1) / float(max_steps)
        action = (1.0 - alpha) * a_from + alpha * a_to

        obs, reward, terminated, truncated, _ = env.step(
            action.squeeze(0).cpu().numpy()
        )
        total_reward += reward

        if step % 10 == 0:
            print(f"[{label}] step={step}, alpha={alpha:.2f}, reward={reward:.3f}")

        if terminated or truncated:
            print(f"[{label}] episode ended early at step {step}")
            done = True
            break

    return obs, total_reward, done


if __name__ == "__main__":
    args = parse_args_ppo()
    device = torch.device("cuda" if args.cuda else "cpu")

    # ===== settings =====
    fps = 30
    walk_duration_sec = 5.0         # walk-forward phase
    transition_duration_sec = 1.0   # smooth blend phase (walk -> turn)
    turn_duration_sec = 5.0         # turn-right phase
    base_env_id = "Humanoid-v5"     # single env for continuous episode
    # ====================

    # ===== prepare video output =====
    video_folder = "./videos_eval"
    os.makedirs(video_folder, exist_ok=True)
    video_name = "ppo_chain_walkforward_then_turnright.mp4"
    video_path = os.path.join(video_folder, video_name)
    # ================================

    # ===== 1) make ONE env for the whole episode =====
    env = make_env_for_video(base_env_id, reward_scale=args.reward_scale, fps=fps)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ===== 2) build two agents: walk-forward and turn-right =====
    # Walk-forward agent
    agent_walk = PPOAgent(obs_dim, action_dim).to(device)
    ckpt_walk_path = os.path.join("ckpts", "wak_forward", "best.pt")
    print(f"[INFO] Loading walk-forward checkpoint from: {ckpt_walk_path}")
    ckpt_walk = torch.load(ckpt_walk_path, map_location=device, weights_only=False)
    agent_walk.load_state_dict(
        ckpt_walk["state_dict"] if isinstance(ckpt_walk, dict) and "state_dict" in ckpt_walk else ckpt_walk
    )
    agent_walk.eval()

    # Turn-right agent
    agent_turn = PPOAgent(obs_dim, action_dim).to(device)
    ckpt_turn_path = os.path.join("ckpts", "turn_right", "best.pt")
    print(f"[INFO] Loading turn-right checkpoint from: {ckpt_turn_path}")
    ckpt_turn = torch.load(ckpt_turn_path, map_location=device, weights_only=False)
    agent_turn.load_state_dict(
        ckpt_turn["state_dict"] if isinstance(ckpt_turn, dict) and "state_dict" in ckpt_turn else ckpt_turn
    )
    agent_turn.eval()

    # ===== 3) reset env once and init video writer =====
    obs, _ = env.reset()
    frame = env.render()
    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    # write the very first frame
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # ===== 4) segments in ONE continuous episode: WALK -> TRANSITION -> TURN =====
    print("\n=== Segment 1: WALK FORWARD ===")
    obs, r_walk, done = run_segment(
        env, agent_walk, writer, device, fps,
        duration_sec=walk_duration_sec,
        obs=obs,
        label="walk_forward"
    )

    r_trans = 0.0
    r_turn = 0.0

    if not done:
        print("\n=== Segment 1.5: TRANSITION (blend walk -> turn) ===")
        obs, r_trans, done = run_transition(
            env, agent_walk, agent_turn, writer, device, fps,
            duration_sec=transition_duration_sec,
            obs=obs,
            label="transition"
        )

    if not done:
        print("\n=== Segment 2: TURN RIGHT ===")
        obs, r_turn, done = run_segment(
            env, agent_turn, writer, device, fps,
            duration_sec=turn_duration_sec,
            obs=obs,
            label="turn_right"
        )

    # ===== 5) cleanup =====
    writer.release()
    env.close()

    print("\n=== Chain finished ===")
    print(f"  Walk-forward reward:   {r_walk:.2f}")
    print(f"  Transition reward:     {r_trans:.2f}")
    print(f"  Turn-right reward:     {r_turn:.2f}")
    print(f"Video saved at: {video_path}")
