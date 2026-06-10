"""
Comparative RL Agents on MineRL: DQN vs PPO for Sparse-Reward Object Collection
DQN Training Script
"""

import os
import gc
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import minerl

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────
os.environ['DISPLAY'] = ':0'
os.environ['MALMO_MINECRAFT_OUTPUT_LOG'] = 'true'
os.environ['MALMO_MINECRAFT_INITIAL_MEMORY'] = '2G'
os.environ['MALMO_MINECRAFT_MAX_MEMORY'] = '4G'
os.environ['MINERL_HEADLESS'] = '1'
os.environ['MINERL_DISABLE_HUB'] = '1'
os.environ['MALMO_MINECRAFT_JVM_ARGS'] = '-XX:+UseG1GC -XX:MaxGCPauseMillis=50'

# ─────────────────────────────────────────────────────────────────────────────
# Replay Memory
# ─────────────────────────────────────────────────────────────────────────────
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ─────────────────────────────────────────────────────────────────────────────
# Environment Wrappers
# ─────────────────────────────────────────────────────────────────────────────
class LogOnlyWrapper(gym.Wrapper):
    """Reward agent only for collecting logs; terminates episode on collection."""

    def __init__(self, env, action_repeat=4):
        super().__init__(env)
        self.logs_collected = 0
        self.action_repeat = action_repeat

    def reset(self):
        self.logs_collected = 0
        return super().reset()

    def step(self, action):
        total_reward = 0
        done = False
        info = None

        for _ in range(self.action_repeat):
            obs, reward, done, info = super().step(action)
            total_reward += reward

            logs_current = self._get_log_count(obs)
            logs_collected_step = max(0, logs_current - self.logs_collected)
            self.logs_collected = logs_current

            if logs_collected_step > 0:
                total_reward = 1.0
                done = True
                break

            if done:
                break

        return obs, total_reward, done, info

    def _get_log_count(self, obs):
        if 'inventory' in obs:
            return obs['inventory'].get('log', 0)
        return 0


class ResizeObservationWrapper(gym.ObservationWrapper):
    """Resize POV observations to a fixed spatial size."""

    def __init__(self, env, size=(64, 64)):
        super().__init__(env)
        self.size = size
        import cv2
        self.cv2 = cv2

    def observation(self, observation):
        if isinstance(observation, dict) and 'pov' in observation:
            original_pov = observation['pov']
            resized_pov = self.cv2.resize(
                original_pov,
                (self.size[1], self.size[0]),
                interpolation=self.cv2.INTER_AREA,
            )
            observation['pov'] = resized_pov
        return observation


class AlwaysAttackWrapper(gym.ActionWrapper):
    """Discrete action space that always has attack enabled."""

    def __init__(self, env, angle=10):
        super().__init__(env)
        self.action_dict = {
            0: {'attack': 1, 'back': 0, 'camera': [0, 0],      'forward': 1, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            1: {'attack': 1, 'back': 0, 'camera': [0, angle],  'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            2: {'attack': 1, 'back': 0, 'camera': [0, -angle], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            3: {'attack': 1, 'back': 0, 'camera': [angle, 0],  'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            4: {'attack': 1, 'back': 0, 'camera': [-angle, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            5: {'attack': 1, 'back': 0, 'camera': [0, 0],      'forward': 1, 'jump': 1, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            6: {'attack': 1, 'back': 0, 'camera': [0, 0],      'forward': 1, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            7: {'attack': 1, 'back': 0, 'camera': [0, 0],      'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
        }
        self.action_space = gym.spaces.Discrete(len(self.action_dict))

    def action(self, action_idx):
        return self.action_dict[action_idx]


# ─────────────────────────────────────────────────────────────────────────────
# DQN Model
# ─────────────────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_output(self, shape):
        batch_size = 1
        input_tensor = torch.zeros(batch_size, *shape)
        output = self.conv(input_tensor)
        return int(np.prod(output.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_observation(observation):
    if isinstance(observation, dict) and 'pov' in observation:
        pov = observation['pov']
        pov = np.transpose(pov, (2, 0, 1))
        pov = pov.astype(np.float32) / 255.0
        return pov
    raise ValueError("Expected observation to contain 'pov' key")


def select_action(state, policy_net, action_space, epsilon, device):
    if random.random() < epsilon:
        return random.randint(0, action_space - 1)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(state_tensor)
        return q_values.max(1)[1].item()


def optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    if len(memory) < batch_size:
        return 0.0

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool,
    )
    non_final_next_states = torch.tensor(
        [s for s in batch.next_state if s is not None],
        device=device, dtype=torch.float32,
    )

    state_batch  = torch.tensor(batch.state,  device=device, dtype=torch.float32)
    action_batch = torch.tensor(batch.action, device=device, dtype=torch.long)
    reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float32)

    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = reward_batch + (gamma * next_state_values)

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item()


def moving_average(data, window_size=50):
    data = np.array(data)
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
    return smooth_data[: -window_size + 1]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_metrics(metrics, save_path='training_metrics.png'):
    plt.figure(figsize=(15, 12))

    plt.subplot(3, 1, 1)
    plt.plot(metrics['episode_rewards'], label='episode_rewards')
    plt.plot(moving_average(metrics['episode_rewards']), label='moving_average')
    plt.title('Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(metrics['episode_durations'], label='episode_durations')
    plt.plot(moving_average(metrics['episode_durations']), label='moving_average')
    plt.title('Episode Duration')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(metrics['episode_losses'], label='episode_losses')
    plt.plot(moving_average(metrics['episode_losses']), label='moving_average')
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training metrics saved to {save_path}")


def plot_evaluation_metrics(metrics, save_path='evaluation_metrics.png'):
    plt.figure(figsize=(15, 12))

    plt.subplot(2, 1, 1)
    plt.plot(metrics['episode_rewards'], label='episode_rewards')
    plt.plot(moving_average(metrics['episode_rewards']), label='moving_average')
    plt.title('Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(metrics['episode_durations'], label='episode_durations')
    plt.plot(moving_average(metrics['episode_durations']), label='moving_average')
    plt.title('Episode Duration')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Evaluation metrics saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def build_env(env_name):
    env = gym.make(env_name)
    env = ResizeObservationWrapper(env, size=(64, 64))
    env = LogOnlyWrapper(env)
    env = AlwaysAttackWrapper(env, angle=10)
    return env


def train_dqn(
    env_name,
    total_timesteps,
    batch_size,
    gamma,
    eps_start,
    eps_end,
    eps_decay,
    target_update_freq,
    learning_rate,
    memory_size,
    eval_freq,
    log_freq,
    start_timestep,
    policy_net,
    target_net,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    watchdog_timeout = 180  # seconds

    env = build_env(env_name)
    observation = env.reset()
    pov = preprocess_observation(observation)
    input_shape = pov.shape
    n_actions = env.action_space.n
    print(f"Observation shape: {input_shape}, Action space size: {n_actions}")

    if policy_net is None:
        policy_net = DQN(input_shape, n_actions).to(device)
    if target_net is None:
        target_net = DQN(input_shape, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_size)

    episode_durations = []
    episode_rewards = []
    episode_losses = []
    episode_count = 0

    current_episode_reward = 0
    current_episode_loss = 0
    current_episode_steps = 0

    if start_timestep > 0:
        epsilon = eps_end + (eps_start - eps_end) * np.exp(-start_timestep / (total_timesteps * 0.5))
    else:
        epsilon = eps_start

    print("Starting training...")
    start_time = time.time()

    state = pov
    os.makedirs("checkpoints/dqn_250k", exist_ok=True)

    timestep = start_timestep
    done = False
    last_action_time = time.time()

    while timestep < total_timesteps:
        current_time = time.time()
        if current_time - last_action_time > watchdog_timeout:
            print("\nWatchdog timer expired — environment appears stuck. Resetting...")
            try:
                env.close()
            except Exception:
                pass
            os.system("pkill -f Minecraft")
            os.system("pkill -f java")
            time.sleep(5)

            env = build_env(env_name)
            observation = env.reset()
            state = preprocess_observation(observation)
            current_episode_reward = 0
            current_episode_loss = 0
            current_episode_steps = 0
            done = False
            last_action_time = time.time()
            continue

        action_idx = select_action(state, policy_net, n_actions, epsilon, device)
        observation, reward, done, _ = env.step(action_idx)

        current_episode_reward += reward
        next_state = None if done else preprocess_observation(observation)

        memory.push(state, action_idx, next_state, reward, done)
        state = next_state

        loss = optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma, device)
        if loss:
            current_episode_loss += loss

        timestep += 1
        current_episode_steps += 1
        last_action_time = time.time()

        if timestep % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print("Target network updated!")

        epsilon = eps_end + (eps_start - eps_end) * np.exp(-timestep / (total_timesteps * 0.5))

        if timestep % log_freq == 0:
            avg_reward   = np.mean(episode_rewards[-10:])   if episode_rewards   else 0
            avg_duration = np.mean(episode_durations[-10:]) if episode_durations else 0
            avg_loss     = np.mean(episode_losses[-10:])    if episode_losses    else 0
            elapsed = time.time() - start_time
            print(
                f"\nTimestep: {timestep}/{total_timesteps} "
                f"({(timestep / total_timesteps) * 100:.1f}%) | "
                f"Episodes: {episode_count} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Avg Reward: {avg_reward:.4f} | "
                f"Avg Duration: {avg_duration:.1f} steps | "
                f"Avg Loss: {avg_loss:.6f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        if timestep % eval_freq == 0 and timestep > 0:
            print("\nSaving checkpoint...")
            checkpoint = {
                'timestep': timestep,
                'policy_net': policy_net.state_dict(),
                'target_net': target_net.state_dict(),
                'epsilon': epsilon,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'episode_durations': episode_durations,
                'episode_rewards': episode_rewards,
                'episode_losses': episode_losses,
                'total_episodes': episode_count,
            }
            checkpoint_path = f"checkpoints/dqn_250k/dqn_step_{timestep}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        if done or current_episode_steps >= 1000:
            episode_count += 1
            episode_durations.append(current_episode_steps)
            episode_rewards.append(current_episode_reward)
            episode_losses.append(
                current_episode_loss / current_episode_steps if current_episode_steps > 0 else 0
            )

            observation = env.reset()
            state = preprocess_observation(observation)
            current_episode_reward = 0
            current_episode_loss = 0
            current_episode_steps = 0
            done = False

            if episode_count % 2 == 0:
                try:
                    env.close()
                    time.sleep(2)
                    os.system("pkill -f Minecraft")
                    env = build_env(env_name)
                except Exception as e:
                    print(f"Error during environment reset: {e}")

            observation = env.reset()

        if timestep % 500 == 0:
            gc.collect()

    env.close()
    end_time = time.time()
    print(f"\nTraining complete! Time elapsed: {end_time - start_time:.2f}s")
    print(f"Total episodes completed: {episode_count}")

    os.makedirs("checkpoints/dqn_50k", exist_ok=True)
    final_checkpoint = {
        'timestep': timestep,
        'policy_net': policy_net.state_dict(),
        'target_net': target_net.state_dict(),
        'epsilon': epsilon,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'episode_durations': episode_durations,
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'total_episodes': episode_count,
    }
    model_path = "checkpoints/dqn_50k/dqn_final.pt"
    torch.save(final_checkpoint, model_path)
    print(f"Final model saved to {model_path}")

    return {
        'policy_net': policy_net,
        'target_net': target_net,
        'episode_durations': episode_durations,
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'total_episodes': episode_count,
        'total_timesteps': timestep,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, env_name, num_episodes=10, angle=10):
    env = build_env(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episode_rewards = []
    episode_durations = []
    success_rate = 0

    for episode in range(num_episodes):
        observation = env.reset()
        state = preprocess_observation(observation)

        total_reward = 0
        step_count = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action_idx = q_values.max(1)[1].item()

            observation, reward, done, _ = env.step(action_idx)
            total_reward += reward
            step_count += 1

            if not done:
                state = preprocess_observation(observation)

            if step_count >= 1000:
                break

        episode_rewards.append(total_reward)
        episode_durations.append(step_count)
        if total_reward > 0:
            success_rate += 1

        print(f"Evaluation Episode {episode + 1}/{num_episodes} | "
              f"Reward: {total_reward:.2f} | Steps: {step_count}")

    avg_reward = np.mean(episode_rewards)
    avg_duration = np.mean(episode_durations)
    success_percentage = (success_rate / num_episodes) * 100

    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Duration: {avg_duration:.2f} steps")
    print(f"Success Rate: {success_percentage:.2f}%")

    return {
        'avg_reward': avg_reward,
        'avg_duration': avg_duration,
        'success_rate': success_percentage,
        'episode_rewards': episode_rewards,
        'episode_durations': episode_durations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loader
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    return {
        'policy_net':        checkpoint['policy_net'],
        'target_net':        checkpoint['target_net'],
        'episode_durations': checkpoint['episode_durations'],
        'episode_rewards':   checkpoint['episode_rewards'],
        'episode_losses':    checkpoint['episode_losses'],
        'total_episodes':    checkpoint['total_episodes'],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ENV_NAME       = "MineRLObtainDiamondShovel-v0"
    TOTAL_TIMESTEPS = 250_000

    training_metrics = train_dqn(
        env_name        = ENV_NAME,
        total_timesteps = TOTAL_TIMESTEPS,
        batch_size      = 64,
        gamma           = 0.99,
        eps_start       = 1.0,
        eps_end         = 0.05,
        eps_decay       = 0.9998,
        target_update_freq = 2500,
        learning_rate   = 0.0001,
        memory_size     = 25000,
        eval_freq       = 1000,
        log_freq        = 1000,
        start_timestep  = 0,
        policy_net      = None,
        target_net      = None,
    )

    plot_training_metrics(training_metrics, save_path='results/figures/DQN_training_metrics.png')

    # Evaluate
    policy_net = training_metrics['policy_net']
    eval_metrics = evaluate_model(policy_net, ENV_NAME, num_episodes=5)
    plot_evaluation_metrics(eval_metrics, save_path='results/figures/DQN_evaluation_metrics.png')
