#!/bin/bash

set -e

echo "=== [1] Load bash config & conda ==="
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda command not found. Are you on the right AWS image?"
    exit 1
fi

echo "=== [2] Ensure conda env 'pytorch_p310' exists ==="
if conda env list | grep -q "pytorch_p310"; then
    echo "Conda env 'pytorch_p310' already exists."
else
    echo "Creating conda env 'pytorch_p310' with Python 3.10..."
    conda create -y -n pytorch_p310 python=3.10
fi

echo "Activating env 'pytorch_p310'..."
conda activate pytorch_p310

echo "=== [3] Upgrade pip tooling in this env ==="
python -m pip install --upgrade pip setuptools wheel

echo "=== [4] Set up MuJoCo directories & environment variables ==="
mkdir -p "$HOME/.mujoco/plugins"

grep -q 'MUJOCO_PATH=' "$HOME/.bashrc" || echo 'export MUJOCO_PATH="$HOME/.mujoco"' >> "$HOME/.bashrc"
grep -q 'MUJOCO_PLUGIN_PATH=' "$HOME/.bashrc" || echo 'export MUJOCO_PLUGIN_PATH="$HOME/.mujoco/plugins"' >> "$HOME/.bashrc"
grep -q 'MUJOCO_GL=' "$HOME/.bashrc" || echo 'export MUJOCO_GL="egl"' >> "$HOME/.bashrc"

export MUJOCO_PATH="$HOME/.mujoco"
export MUJOCO_PLUGIN_PATH="$HOME/.mujoco/plugins"
export MUJOCO_GL="egl"

echo "MUJOCO_PATH set to: $MUJOCO_PATH"
echo "MUJOCO_PLUGIN_PATH set to: $MUJOCO_PLUGIN_PATH"
echo "MUJOCO_GL set to: $MUJOCO_GL"

echo "=== [5] Install Python packages (idempotent) ==="
python - << 'EOF'
import importlib, sys
spec = importlib.util.find_spec("torch")
if spec is None:
    sys.exit(1)
print("PyTorch already installed.")
EOF

if [ $? -ne 0 ]; then
    echo "Installing PyTorch..."
    python -m pip install torch
fi

echo "Installing RL & MuJoCo dependencies..."
python -m pip install \
    "mujoco==3.3.4" \
    "gymnasium[mujoco]" \
    gymnasium \
    scikit-image \
    tensorboardX \
    tqdm \
    imageio \
    imageio-ffmpeg \
    moviepy

echo "=== [6] Navigate to PPO-Humanoid project directory ==="
PROJECT_DIR="$HOME/SageMaker/new_robotics/PPO-Humanoid"
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    echo "Now in: $(pwd)"
else
    echo "WARNING: Project dir '$PROJECT_DIR' not found."
fi

echo "=== [7] Quick Humanoid-v5 sanity check ==="
python - << 'EOF'
import gymnasium as gym

try:
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    obs, info = env.reset()
    frame = env.render()
    print("[OK] Humanoid-v5 created.")
    print("     obs shape  :", getattr(obs, "shape", type(obs)))
    print("     frame shape:", getattr(frame, "shape", type(frame)))
    env.close()
except Exception as e:
    print("[WARN] Humanoid-v5 test failed:")
    print("       ", e)
EOF

echo "=== [8] DONE ==="
echo "Use:"
echo "    conda activate pytorch_p310"
echo "    cd $PROJECT_DIR"
echo "    python test_ppo.py   # or trainer.py"
