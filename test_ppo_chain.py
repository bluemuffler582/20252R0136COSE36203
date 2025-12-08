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
    make_env logic as training (but with render=True).
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
            obs_tensor = torch.tensor(
                np.array([obs], dtype=np.float32), device=device
            )
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


def run_transition(
    env,
    agent_from,
    agent_to,
    writer,
    device,
    fps,
    duration_sec,
    obs,
    label="transition",
    alpha_max=1.0,
):
    """
    Smoothly transition from agent_from to agent_to over `duration_sec`.

    The blended action is:
        a = (1 - alpha) * a_from + alpha * a_to

    where alpha goes from 0 -> alpha_max (<= 1) using cosine easing.
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
            obs_tensor = torch.tensor(
                np.array([obs], dtype=np.float32), device=device
            )
            a_from, _, _, _ = agent_from.get_action_and_value(obs_tensor)
            a_to,   _, _, _ = agent_to.get_action_and_value(obs_tensor)

        # cosine-eased alpha: 0 -> 1, then scaled to 0 -> alpha_max
        t = float(step + 1) / float(max_steps)      # 0..1
        alpha = 0.5 * (1.0 - np.cos(np.pi * t))     # 0..1
        alpha = alpha_max * alpha                   # 0..alpha_max

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

    # durations (you can tune these)
    walk_fwd_duration_sec = 5.0        # walk-forward phase
    balance_duration_sec   = 5.0        # pure balance in the middle
    walk_bwd_duration_sec  = 5.0        # walk-backward phase

    transition_walk_to_bal_sec = 5.0    # blend walk -> balance
    transition_bal_to_bwd_sec  = 5.0    # blend balance -> backward

    # how strongly to mix balance in the first transition
    alpha_max_walk_to_bal = 0.5   # 1.0 = fully hand control to balance by end
    alpha_max_bal_to_bwd  = 0.5   # fully hand to backward by end

    # Use the CLI env as the base env (normally Humanoid-v5)
    base_env_id = args.env
    # ====================

    # ===== prepare video output =====
    video_folder = "./videos_eval"
    os.makedirs(video_folder, exist_ok=True)
    video_name = "ppo_chain_walk_forward_balance_backward.mp4"
    video_path = os.path.join(video_folder, video_name)
    # ================================

    # ===== 1) make ONE env for the whole episode =====
    env = make_env_for_video(base_env_id, reward_scale=args.reward_scale, fps=fps)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ===== 2) build three agents: walk-forward, balance, walk-backward =====
    # Walk-forward agent
    agent_walk_fwd = PPOAgent(obs_dim, action_dim).to(device)
    ckpt_walk_fwd_path = os.path.join("ckpts", "walk_forward", "best.pt")
    print(f"[INFO] Loading walk-forward checkpoint from: {ckpt_walk_fwd_path}")
    ckpt_walk_fwd = torch.load(ckpt_walk_fwd_path, map_location=device, weights_only=False)
    agent_walk_fwd.load_state_dict(
        ckpt_walk_fwd["state_dict"]
        if isinstance(ckpt_walk_fwd, dict) and "state_dict" in ckpt_walk_fwd
        else ckpt_walk_fwd
    )
    agent_walk_fwd.eval()

    # Balance agent
    agent_balance = PPOAgent(obs_dim, action_dim).to(device)
    ckpt_balance_path = os.path.join("ckpts", "balance", "best.pt")
    print(f"[INFO] Loading balance checkpoint from: {ckpt_balance_path}")
    ckpt_balance = torch.load(ckpt_balance_path, map_location=device, weights_only=False)
    agent_balance.load_state_dict(
        ckpt_balance["state_dict"]
        if isinstance(ckpt_balance, dict) and "state_dict" in ckpt_balance
        else ckpt_balance
    )
    agent_balance.eval()

    # Walk-backward agent
    agent_walk_bwd = PPOAgent(obs_dim, action_dim).to(device)
    ckpt_walk_bwd_path = os.path.join("ckpts", "walk_backward", "best.pt")
    print(f"[INFO] Loading walk-backward checkpoint from: {ckpt_walk_bwd_path}")
    ckpt_walk_bwd = torch.load(ckpt_walk_bwd_path, map_location=device, weights_only=False)
    agent_walk_bwd.load_state_dict(
        ckpt_walk_bwd["state_dict"]
        if isinstance(ckpt_walk_bwd, dict) and "state_dict" in ckpt_walk_bwd
        else ckpt_walk_bwd
    )
    agent_walk_bwd.eval()

    # ===== 3) reset env once and init video writer =====
    obs, _ = env.reset()
    frame = env.render()
    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    # write the very first frame
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # ===== 4) segments in ONE continuous episode =====

    # ---- Segment 1: WALK FORWARD ----
    print("\n=== Segment 1: WALK FORWARD ===")
    obs, r_walk_fwd, done = run_segment(
        env, agent_walk_fwd, writer, device, fps,
        duration_sec=walk_fwd_duration_sec,
        obs=obs,
        label="walk_forward"
    )

    r_trans_w2b = 0.0
    r_balance   = 0.0
    r_trans_b2w = 0.0
    r_walk_bwd  = 0.0

    # ---- Transition 1: WALK FORWARD -> BALANCE (alpha blend) ----
    if not done:
        print("\n=== Segment 1.5: TRANSITION (walk_forward -> balance) ===")
        obs, r_trans_w2b, done = run_transition(
            env,
            agent_walk_fwd,
            agent_balance,
            writer,
            device,
            fps,
            duration_sec=transition_walk_to_bal_sec,
            obs=obs,
            label="transition_walk_to_balance",
            alpha_max=alpha_max_walk_to_bal,   # 1.0 = fully balance at end
        )

    # ---- Segment 2: PURE BALANCE (no blending) ----
    if not done:
        print("\n=== Segment 2: BALANCE (pure) ===")
        obs, r_balance, done = run_segment(
            env, agent_balance, writer, device, fps,
            duration_sec=balance_duration_sec,
            obs=obs,
            label="balance"
        )

    # ---- Transition 2: BALANCE -> WALK BACKWARD (alpha blend) ----
    if not done:
        print("\n=== Segment 2.5: TRANSITION (balance -> walk_backward) ===")
        obs, r_trans_b2w, done = run_transition(
            env,
            agent_balance,
            agent_walk_bwd,
            writer,
            device,
            fps,
            duration_sec=transition_bal_to_bwd_sec,
            obs=obs,
            label="transition_balance_to_backward",
            alpha_max=alpha_max_bal_to_bwd,  # usually 1.0
        )

    # ---- Segment 3: WALK BACKWARD ----
    if not done:
        print("\n=== Segment 3: WALK BACKWARD ===")
        obs, r_walk_bwd, done = run_segment(
            env, agent_walk_bwd, writer, device, fps,
            duration_sec=walk_bwd_duration_sec,
            obs=obs,
            label="walk_backward"
        )

    # ===== 5) cleanup =====
    writer.release()
    env.close()

    print("\n=== Chain finished ===")
    print(f"  Walk-forward reward:            {r_walk_fwd:.2f}")
    print(f"  Transition (walk->balance):     {r_trans_w2b:.2f}")
    print(f"  Balance (pure) reward:          {r_balance:.2f}")
    print(f"  Transition (balance->backward): {r_trans_b2w:.2f}")
    print(f"  Walk-backward reward:           {r_walk_bwd:.2f}")
    print(f"Video saved at: {video_path}")
