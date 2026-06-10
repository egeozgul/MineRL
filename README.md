# Efficient Item Collection in Minecraft

Reinforcement learning agents for collecting wood logs in Minecraft using the [MineRL](https://minerl.io/) benchmark. This project implements and compares **DQN** and **PPO** on `MineRLObtainDiamondShovel-v0`, with a focus on sample efficiency, training stability, and reward shaping for log collection.

## Overview

| Algorithm | Implementation | Notes |
|-----------|----------------|-------|
| DQN | [`notebooks/train_dqn.ipynb`](notebooks/train_dqn.ipynb) | Custom CNN, experience replay, epsilon-greedy exploration |
| PPO | [`scripts/train_ppo.py`](scripts/train_ppo.py) | Stable-Baselines3 with simplified discrete action space |

Both agents use POV (pixel) observations resized to 64×64, a reduced action space (10 discrete actions), and custom reward shaping that rewards log collection.

## Architecture

### DQN

Experience replay with a periodically synced target network for stable Q-value learning.

![DQN architecture](docs/diagrams/dqn_architecture.png)

### PPO

Shared CNN backbone with separate actor (policy) and critic (value) heads.

![PPO architecture](docs/diagrams/ppo_architecture.png)

## Results

| DQN training (50k steps) | DQN training (250k steps) | PPO evaluation |
|---|---|---|
| ![DQN 50k](results/figures/training_metrics_50k.png) | ![DQN 250k](results/figures/training_metrics_250k.png) | ![PPO eval](results/figures/evaluation_metrics_50k.png) |

Full write-up: [`docs/final_report.pdf`](docs/final_report.pdf)

## Project Structure

```
.
├── notebooks/
│   └── train_dqn.ipynb          # DQN training and evaluation
├── scripts/
│   └── train_ppo.py             # PPO training and evaluation
├── checkpoints/
│   ├── dqn_50k/                 # DQN weights (50k timesteps)
│   └── dqn_250k/                # DQN weights (250k timesteps)
├── results/
│   ├── figures/                 # Training / evaluation plots
│   └── ppo/                     # PPO training logs
├── docs/
│   ├── diagrams/                # Model architecture diagrams
│   ├── final_report.pdf
│   └── project_proposal.pdf
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- Java 8 (required by MineRL / Malmo)
- CUDA-capable GPU recommended (CPU training is supported but slow)

## Installation

```bash
git clone https://github.com/egeozgul/MineRL.git
cd MineRL

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install minerl
```

> **Note:** MineRL requires additional system setup (Java, display/headless config). See the [official docs](https://minerl.io/docs/) if environment creation fails.

## Usage

### DQN

Open and run [`notebooks/train_dqn.ipynb`](notebooks/train_dqn.ipynb). Pre-trained weights are in [`checkpoints/`](checkpoints/).

### PPO

```bash
python scripts/train_ppo.py
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOTAL_TIMESTEPS` | 1000 | Training timesteps |
| `TOTAL_EPISODES` | 50 | Evaluation episodes after training |
| `MAX_STEPS_PER_EPISODE` | 1000 | Episode length cap |
| `SHOWFRAMES` | `False` | Save episode frames to disk |
| `LEARNING_RATE` | `1e-3` | PPO learning rate |

## Demo Video

[![MineRL DQN Agent](https://github.com/user-attachments/assets/62f2840a-2c18-41bb-a013-6b0cf39372be)](https://drive.google.com/file/d/1ijB8jZFLVUQXNKsgVVMk-LmDZeFx2NTq/view?usp=drive_link)

## Author

[Ege Ozgul](https://github.com/egeozgul)

## Citation

```bibtex
@misc{Ozgul2025minerl,
  title={Efficient Item Collection in Minecraft},
  author={Ege Ozgul},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/egeozgul/MineRL}}
}
```
