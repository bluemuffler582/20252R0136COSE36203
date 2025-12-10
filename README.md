✅ AWS Humanoid RL Environment Setup (Clean + Organized Guide)
0. Make sure you're using Bash
(If already in bash, this is harmless.)
[ "$0" != "bash" ] && exec bash
1. Initialize and Activate Conda
# Initialize conda (only needed once)
conda init

# Reload shell (optional)
source ~/.bashrc

# Activate your DL environment
conda activate pytorch_p310

# Check Python version
python -V   # should show 3.10.x
2. Install Core Packages (PyTorch, Mujoco, Gymnasium, TensorboardX)
Upgrade pip
python -m pip install --upgrade pip
Install PyTorch (CUDA 11.8, best for AWS DLAMI)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Install MuJoCo + Gymnasium
python -m pip install "mujoco==3.1.6" "gymnasium[mujoco]"
Install tensorboardX
python -m pip install tensorboardX
3. MuJoCo Environment Variables (Session Only)
export MUJOCO_GL=egl
export MUJOCO_PLUGIN_PATH=$HOME/.mujoco/mujoco-plugins
mkdir -p "$MUJOCO_PLUGIN_PATH"

# If you're also using mujoco210 binaries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/.mujoco/mujoco210/bin
4. Make MuJoCo Variables Persistent (Add to ~/.bashrc)
grep -qxF 'export MUJOCO_GL=egl' ~/.bashrc || echo 'export MUJOCO_GL=egl' >> ~/.bashrc
grep -qxF 'export MUJOCO_PLUGIN_PATH=$HOME/.mujoco/mujoco-plugins' ~/.bashrc || echo 'export MUJOCO_PLUGIN_PATH=$HOME/.mujoco/mujoco-plugins' >> ~/.bashrc
grep -qxF 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/.mujoco/mujoco210/bin' ~/.bashrc || echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/.mujoco/mujoco210/bin' >> ~/.bashrc
5. Sanity Check That Everything Imports Correctly
python - << 'EOF'
import torch
import gymnasium as gym
import mujoco
print("Torch  version:", torch.__version__)
print("Gym    version:", gym.__version__)
print("Mujoco version:", mujoco.__version__)
EOF
6. Install LLM Dependencies (Transformers, Datasets, etc.)
Navigate to your project:
cd ~/SageMaker/new_robotics/PPO-Humanoid
Install all the needed Python packages:
pip install \
    "transformers>=4.30.0" \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0" \
    "tqdm>=4.65.0" \
    tokenizers \
    accelerate \
    safetensors \
    datasets
