# AWS Humanoid RL Environment Setup

This guide explains how to configure an AWS DLAMI (Deep Learning AMI) instance for running Humanoid Reinforcement Learning with MuJoCo, Gymnasium, PyTorch, and an LLM-based skill router.

## 0. Ensure You’re Using Bash

```
[ "$0" != "bash" ] && exec bash
```

## 1. Initialize and Activate Conda

### Initialize conda (only needed once)
```
conda init
```

### Reload shell
```
source ~/.bashrc
```

### Activate your DL environment
```
conda activate pytorch_p310
```

### Check Python version
```
python -V
# Expected output: Python 3.10.x
```

## 2. Install Core Packages

### Upgrade pip
```
python -m pip install --upgrade pip
```

### Install PyTorch (CUDA 11.8 build for AWS)
```
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install MuJoCo + Gymnasium
```
python -m pip install "mujoco==3.1.6" "gymnasium[mujoco]"
```

### Install TensorboardX
```
python -m pip install tensorboardX
```

## 3. Configure MuJoCo Environment Variables (Session Only)

```
export MUJOCO_GL=egl
export MUJOCO_PLUGIN_PATH=$HOME/.mujoco/mujoco-plugins
mkdir -p "$MUJOCO_PLUGIN_PATH"

# If using mujoco210 binaries (optional)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/.mujoco/mujoco210/bin
```

## 4. Make MuJoCo Variables Persistent

```
grep -qxF 'export MUJOCO_GL=egl' ~/.bashrc || echo 'export MUJOCO_GL=egl' >> ~/.bashrc
grep -qxF 'export MUJOCO_PLUGIN_PATH=$HOME/.mujoco/mujoco-plugins' ~/.bashrc || echo 'export MUJOCO_PLUGIN_PATH=$HOME/.mujoco/mujoco-plugins' >> ~/.bashrc
grep -qxF 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/.mujoco/mujoco210/bin' ~/.bashrc || echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/.mujoco/mujoco210/bin' >> ~/.bashrc
```

## 5. Sanity Check

```
python - << 'EOF'
import torch
import gymnasium as gym
import mujoco

print("Torch  version:", torch.__version__)
print("Gym    version:", gym.__version__)
print("Mujoco version:", mujoco.__version__)
EOF
```

## 6. Install LLM Dependencies

```
cd ~/SageMaker/new_robotics/PPO-Humanoid

pip install     "transformers>=4.30.0"     "numpy>=1.24.0"     "pandas>=2.0.0"     "scikit-learn>=1.3.0"     "tqdm>=4.65.0"     tokenizers     accelerate     safetensors     datasets
```

## Setup Complete

You now have a fully working AWS environment ready for PPO humanoid training, MuJoCo simulation, LLM skill routing, and video rendering.
