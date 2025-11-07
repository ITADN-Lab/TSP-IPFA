import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from models import ActorNetwork, CriticNetwork
from memory import PPOMemory
from config import TRAIN_CONFIG


class PPOAgent:
    def __init__(self, input_dims, num_generators, lr=TRAIN_CONFIG['learning_rate'],
                 gamma=TRAIN_CONFIG['gamma'], epsilon=TRAIN_CONFIG['epsilon'],
                 batch_size=TRAIN_CONFIG['batch_size'], n_epochs=TRAIN_CONFIG['n_epochs']):

        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.num_actions = num_generators * 2

        self.reward_window = deque(maxlen=300)
        for _ in range(300):
            self.reward_window.append(0.0)

        self.total_steps = 0
        self.actor = ActorNetwork(input_dims, num_generators)
        self.critic = CriticNetwork(input_dims)
        self.log_std = nn.Parameter(torch.zeros(1, self.num_actions))

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        self.critic_scheduler = optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=50, gamma=0.9
        )

        self.memory = PPOMemory(batch_size=batch_size, state_dim=input_dims, action_dim=self.num_actions)

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float)
        mu = self.actor(state)
        std = torch.exp(self.log_std).squeeze(0)

        progress = min(1.0, self.total_steps / 5000)
        exploration_sigma = 0.4 * (1 - progress) ** 2

        platform_count = getattr(self, 'platform_count', 0)
        if len(self.reward_window) == self.reward_window.maxlen:
            reward_values = list(self.reward_window)
            current_std = np.std(reward_values)
            if current_std < 1.5:
                platform_count += 1
                if platform_count >= 2:
                    exploration_sigma *= 1.8
                    platform_count = 0
            else:
                platform_count = 0
        self.platform_count = platform_count

        total_std = std + exploration_sigma
        dist = torch.distributions.Normal(mu, total_std)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)

        log_prob = dist.log_prob(action).sum()
        val = self.critic(state)

        return action.detach().numpy(), log_prob.detach(), val.detach()

    def store_transition(self, state, action, prob, val, reward, done, phase=1):
        self.memory.store_memory(state, action, prob, val, reward, done, phase)

    def learn(self):
        self.total_steps += 1

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, val_arr, reward_arr, done_arr = \
                self.memory.generate_batches()

            state_arr = np.array(state_arr, dtype=float).reshape(-1, state_arr.shape[-1])
            action_arr = np.array(action_arr, dtype=float).reshape(-1, action_arr.shape[-1])
            old_prob_arr = np.array(old_prob_arr).reshape(-1)
            val_arr = np.array(val_arr).reshape(-1)
            reward_arr = np.array(reward_arr).reshape(-1)
            done_arr = np.array(done_arr).reshape(-1)

            advantages = np.zeros_like(reward_arr, dtype=np.float32)
            returns = np.zeros_like(reward_arr, dtype=np.float32)

            gae = 0
            for t in reversed(range(len(reward_arr))):
                if t == len(reward_arr) - 1:
                    next_value = 0
                else:
                    next_value = val_arr[t + 1]

                delta = reward_arr[t] + self.gamma * next_value * (1 - done_arr[t]) - val_arr[t]
                gae = delta + self.gamma * 0.95 * gae * (1 - done_arr[t])
                advantages[t] = gae
                returns[t] = advantages[t] + val_arr[t]

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            states = torch.tensor(state_arr, dtype=torch.float32)
            old_probs = torch.tensor(old_prob_arr, dtype=torch.float32)
            actions = torch.tensor(action_arr, dtype=torch.float32)
            advantages = torch.tensor(advantages, dtype=torch.float32)
            returns = torch.tensor(returns, dtype=torch.float32)

            try:
                mu = self.actor(states)
                std_init = 0.1
                std_decay = 0.99
                current_std = max(std_init * (std_decay ** self.total_steps), 0.01)
                std = torch.ones_like(mu) * current_std
                std = torch.clamp(std, min=1e-4, max=1.0)

                dist = torch.distributions.Normal(mu, std)
                new_probs = dist.log_prob(actions).sum(1)
                old_probs = old_probs.to(new_probs.device)
                prob_ratio = torch.exp(new_probs - old_probs)

                surr1 = prob_ratio * advantages
                surr2 = torch.clamp(prob_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = torch.clamp(actor_loss, min=-10, max=10)

                entropy = dist.entropy().mean()
                actor_loss -= 0.01 * entropy

                critic_values = self.critic(states).squeeze()
                critic_loss = F.smooth_l1_loss(critic_values, returns)

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=1.0
                )

                if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                    self.actor.apply(self.actor._init_weights)
                    return float('inf'), float('inf')

                self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.actor_scheduler.step(actor_loss)

                return actor_loss.item(), critic_loss.item()

            except Exception as e:
                self.actor.apply(self.actor._init_weights)
                return float('inf'), float('inf')

    def update_phase(self, reward):
        if isinstance(reward, (list, np.ndarray)):
            reward = np.mean(reward)
        elif not isinstance(reward, (int, float)):
            try:
                reward = float(reward)
            except (TypeError, ValueError):
                reward = 0.0

        self.reward_window.append(reward)