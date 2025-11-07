import matplotlib.pyplot as plt
import csv
import json
import os
import numpy as np
from collections import deque
import torch
from config import PLOT_CONFIG


class RewardTracker:
    def __init__(self):
        self.rewards_per_step = []
        self.cumulative_rewards = []

    def add_reward(self, reward):
        if torch.is_tensor(reward):
            self.rewards_per_step.append(reward.item())
        else:
            self.rewards_per_step.append(float(reward))
        self.cumulative_rewards.append(sum(self.rewards_per_step))

    def plot_rewards(self, title="Reward Progression in Epoch", epoch=1):
        plt.rcParams.update(PLOT_CONFIG)
        plt.figure(figsize=(12, 6), dpi=PLOT_CONFIG['dpi'])

        plt.subplot(2, 1, 1)
        plt.plot(self.rewards_per_step, label='Step Rewards', color='blue')
        plt.title(f'{title} - Step Rewards', fontweight='bold')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.cumulative_rewards, label='Cumulative Rewards', color='red')
        plt.title(f'{title} - Cumulative Rewards', fontweight='bold')
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()

        plt.tight_layout()
        os.makedirs('./output', exist_ok=True)
        plt.savefig(f'./output/training_curves{epoch}.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()


class TrainingDataRecorder:
    def __init__(self, output_dir="output", filename="training_data.csv"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.filename = os.path.join(output_dir, filename)
        self.data = []

        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'total_steps', 'avg_reward', 'phase_switch_step'])

    def record_epoch_data(self, epoch, total_steps, avg_reward, phase_switch_step):
        phase_switch_step = phase_switch_step if phase_switch_step is not None else 0
        epoch_data = {
            'epoch': epoch,
            'total_steps': total_steps,
            'avg_reward': avg_reward,
            'phase_switch_step': phase_switch_step
        }
        self.data.append(epoch_data)

        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, total_steps, avg_reward, phase_switch_step])

    def save_summary(self):
        summary = {
            'total_epochs': len(self.data),
            'avg_steps_early': float(np.mean([d['total_steps'] for d in self.data[:200]])),
            'avg_steps_late': float(np.mean([d['total_steps'] for d in self.data[-200:]])),
            'avg_reward_early': float(np.mean([d['avg_reward'] for d in self.data[:200]])),
            'avg_reward_late': float(np.mean([d['avg_reward'] for d in self.data[-200:]])),
            'phase_switch_rate': float(np.mean([1 if d['phase_switch_step'] > 0 else 0 for d in self.data]))
        }

        summary_file = os.path.join(self.output_dir, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary_file