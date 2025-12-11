# Project Structure 

```text
cose362-machine_learning/
│
├── llm/                       # Language model (SkillLanguageModel)
│   ├── data/                  # Dataset
│   │   └── train_data.json    # Training data
│   ├── models/                # Model definitions
│   │   ├── __init__.py
│   │   └── skill_model.py     # SkillLanguageModel implementation
│   ├── checkpoints/           # Saved model weights (auto-generated)
│   ├── dataset.py             # Dataset loader
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── predict.py             # Inference script (text → skill + params)
│   ├── test_model.py          # Dataset/model sanity test
│   ├── requirements.txt       # Dependencies for LLM module
│   └── README.md              # Documentation for LLM module
│
├── llm_results/               # Screenshots and logs of LLM outputs
│
├── rl/                        # Reinforcement Learning code
│   ├── ckpts/                 # Best & last saved PPO weights per skill
│   ├── envs/                  # Custom MuJoCo environments per skill
│   ├── lib/                   # PPO agent, networks, hyperparameters
│   ├── videos_train/          # Videos generated during RL training
│   ├── train_ppo.py           # Train PPO depending on environment
│   ├── test_ppo.py            # Run trained models + save evaluation videos
│   ├── test_ppo_chain.py      # Experimental sequential/multi-skill execution
│   └── __init__.py
│
├── rl_results/                # RL training/evaluation results
│
├── videos_eval/               # Final evaluation videos (.mp4)
│
├── main.py                    # High-level glue: LLM → RL → execution
│
├── demo.mp4                   # Example end-to-end demonstration
│
└── README.md                  # Main documentation
```



# AWS Humanoid RL Environment Setup (Complete Guide)

This guide explains how to configure an AWS DLAMI instance for Humanoid Reinforcement Learning with MuJoCo, Gymnasium, PyTorch, and LLM-based command parsing.

---

## 0. Make sure we're in bash

(If you're already in bash, this is harmless.)

```
[ "$0" != "bash" ] && exec bash
```

---

## 1. Initialize conda & activate environment

### Load conda
```
source /home/ec2-user/anaconda3/bin/activate
```

### Activate your environment
```
conda activate pytorch_p310
```

### Confirm active environment and Python version
```
echo ">> Active env: $CONDA_DEFAULT_ENV"
python -V
```

---

## 2. Install / confirm core packages in THIS env

### Upgrade pip
```
python -m pip install --upgrade pip
```

### Install PyTorch (CUDA 11.8 build, standard for AWS DLAMI)
```
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install MuJoCo + Gymnasium + TensorboardX
```
python -m pip install "mujoco==3.1.6" "gymnasium[mujoco]" tensorboardX
```

---

## 3. Set MuJoCo-related environment variables (current session)

```
export MUJOCO_GL=egl
export MUJOCO_PLUGIN_PATH=$HOME/.mujoco/mujoco-plugins
mkdir -p "$MUJOCO_PLUGIN_PATH"

# If you're using mujoco210 binaries:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/.mujoco/mujoco210/bin
```

---

## 4. Make environment variables persistent for future sessions

```
grep -qxF 'export MUJOCO_GL=egl' ~/.bashrc || echo 'export MUJOCO_GL=egl' >> ~/.bashrc
grep -qxF 'export MUJOCO_PLUGIN_PATH=$HOME/.mujoco/mujoco-plugins' ~/.bashrc || echo 'export MUJOCO_PLUGIN_PATH=$HOME/.mujoco/mujoco-plugins' >> ~/.bashrc
grep -qxF 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/.mujoco/mujoco210/bin' ~/.bashrc || echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/.mujoco/mujoco210/bin' >> ~/.bashrc
```

---

## 5. Quick sanity check: can we import stuff?

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

---

## 6. Go to your project

```
cd ~/SageMaker/new_robotics/PPO-Humanoid
```

---

# LLM Dependencies

### Full installation set

```
pip install     transformers>=4.30.0     numpy>=1.24.0     pandas>=2.0.0     scikit-learn>=1.3.0     tqdm>=4.65.0     tokenizers     accelerate     safetensors
```

### Additional installs (redundant but kept as requested)"::
```
pip install transformers datasets numpy pandas scikit-learn tqdm
pip install tokenizers
pip install accelerate safetensors
```

---

# Setup Complete

Your AWS instance is now ready for PPO training, MuJoCo simulation, LLM parsing, and humanoid skill execution.
