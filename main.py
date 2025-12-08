import json
import subprocess
from typing import Dict, Any


# =========================
# 1. Call LLM (llm/predict.py)
# =========================

def call_llm(text: str) -> Dict[str, Any]:
    """
    Call the language model and return a dict like:
    {
      "text": "...",
      "skill": "walk" | "turn" | "reject" | ...,
      "params": { ... }
    }
    """
    result = subprocess.run(
        ["python", "llm/predict.py", "--text", text],
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout.strip()
    print("\n[LLM raw output]")
    print(stdout)

    data = json.loads(stdout)
    return data


# =========================
# 2. Run the appropriate RL script
# =========================

def run_skill(pred: Dict[str, Any]) -> None:
    skill = pred.get("skill")
    params = pred.get("params", {}) or {}

    print(f"\n[Parsed skill] {skill}, params = {params}")

    # ----- reject / invalid -----
    if skill == "reject":
        print("[INFO] LLM rejected this command (no valid humanoid skill).")
        return

    cmd = None  # final command list we'll pass to subprocess

    # ----- walk -----
    if skill == "walk":
        direction = params.get("direction", "forward")
        steps = params.get("steps", 3)

        # Simple mapping: "steps" -> duration (seconds)
        # You can tune this; for now 1 step ≈ 1s
        duration = float(steps)

        if direction == "forward":
            env_name = "Humanoid-v5"
        elif direction == "backward":
            env_name = "HumanoidWalkBackward-v0"
        else:
            print(f"[WARN] Unknown walk direction: {direction}")
            return

        cmd = [
            "python",
            "rl/test_ppo.py",
            "--env",
            env_name,
            "--duration",
            str(duration),
        ]

    # ----- turn -----
    elif skill == "turn":
        direction = params.get("direction", "left")
        angle = float(params.get("angle", 90))

        # Map angle → duration. Tune this constant as you like.
        # Here: 90 degrees ≈ 2 seconds.
        base_duration_for_90 = 2.0
        duration = base_duration_for_90 * (angle / 90.0)

        if direction == "left":
            env_name = "HumanoidTurnLeft-v0"
        elif direction == "right":
            env_name = "HumanoidTurnRight-v0"
        else:
            print(f"[WARN] Unknown turn direction: {direction}")
            return

        cmd = [
            "python",
            "rl/test_ppo.py",
            "--env",
            env_name,
            "--duration",
            str(duration),
        ]

    # ----- balance (if/when you add it in the LLM) -----
    elif skill == "balance":
        # if LLM ever outputs skill="balance"
        duration = float(params.get("duration", 3.0))
        env_name = "HumanoidBalance-v0"

        cmd = [
            "python",
            "rl/test_ppo.py",
            "--env",
            env_name,
            "--reward-scale",
            "0.01",
            "--duration",
            str(duration),
        ]

    else:
        print(f"[WARN] Unknown skill from LLM: {skill}")
        return

    # Actually run the RL script
    print("\n[EXEC] Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


# =========================
# 3. REPL loop
# =========================

def main():
    print("=== Language → Humanoid controller ===")
    print("Type Korean commands like '앞으로 다섯 걸음 걸어', '왼쪽으로 90도 회전해'")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            text = input("명령: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT]")
            break

        if not text:
            continue

        if text.lower() in {"quit", "exit"}:
            print("[EXIT]")
            break

        try:
            prediction = call_llm(text)
        except Exception as e:
            print(f"[ERROR] Failed to call LLM: {e}")
            continue

        try:
            run_skill(prediction)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] RL script failed: {e}")


if __name__ == "__main__":
    main()
