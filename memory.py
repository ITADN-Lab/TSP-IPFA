import numpy as np
import torch


class PPOMemory:
    def __init__(self, batch_size=64, max_buffer_size=100000, state_dim=50, action_dim=10, decay_factor=0.9):
        self.states = np.zeros((max_buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_buffer_size, action_dim), dtype=np.float32)
        self.probs = np.zeros(max_buffer_size, dtype=np.float32)
        self.vals = np.zeros(max_buffer_size, dtype=np.float32)
        self.rewards = np.zeros(max_buffer_size, dtype=np.float32)
        self.dones = np.zeros(max_buffer_size, dtype=np.bool_)
        self.weights = np.zeros(max_buffer_size, dtype=np.float32)
        self.phases = np.zeros(max_buffer_size, dtype=np.int32)

        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.decay_factor = decay_factor
        self.ptr = 0
        self.total_stored = 0
        self.action_dim = action_dim

    def store_memory(self, state, action, prob, val, reward, done, phase=1):
        prob = max(0, min(1, prob))
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        state = np.asarray(state, dtype=np.float32).flatten()

        if not isinstance(action, np.ndarray):
            try:
                action = np.asarray(action, dtype=np.float32)
            except:
                action = np.array(action, dtype=np.float32).flatten()

        if action.size != self.action_dim:
            if action.size > self.action_dim:
                action = action[:self.action_dim]
            else:
                padded_action = np.zeros(self.action_dim, dtype=np.float32)
                padded_action[:action.size] = action
                action = padded_action

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.probs[self.ptr] = prob
        self.vals[self.ptr] = val
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.phases[self.ptr] = phase
        self.weights[self.ptr] = self.decay_factor ** (self.total_stored - self.ptr)

        self.ptr = (self.ptr + 1) % self.max_buffer_size
        self.total_stored = min(self.total_stored + 1, self.max_buffer_size)

    def generate_batches(self):
        n_states = min(self.total_stored, self.max_buffer_size)

        if n_states <= self.batch_size:
            return self._convert_to_tensors(n_states)

        valid_weights = self.weights[:n_states]
        if np.all(valid_weights == 0):
            probabilities = np.ones(n_states) / n_states
        else:
            probabilities = valid_weights / (valid_weights.sum() + 1e-8)

        try:
            batch_indices = np.random.choice(
                n_states,
                size=min(self.batch_size, n_states),
                p=probabilities,
                replace=False
            )
        except ValueError:
            batch_indices = np.random.choice(
                n_states,
                size=min(self.batch_size, n_states),
                p=probabilities,
                replace=True
            )

        return self._convert_to_tensors(n_states, batch_indices)

    def _convert_to_tensors(self, n_states, batch_indices=None):
        if batch_indices is None:
            return (
                torch.from_numpy(self.states[:n_states]),
                torch.from_numpy(self.actions[:n_states]),
                torch.from_numpy(self.probs[:n_states]),
                torch.from_numpy(self.vals[:n_states]),
                torch.from_numpy(self.rewards[:n_states]),
                torch.from_numpy(self.dones[:n_states])
            )
        else:
            return (
                torch.from_numpy(self.states[batch_indices]),
                torch.from_numpy(self.actions[batch_indices]),
                torch.from_numpy(self.probs[batch_indices]),
                torch.from_numpy(self.vals[batch_indices]),
                torch.from_numpy(self.rewards[batch_indices]),
                torch.from_numpy(self.dones[batch_indices])
            )

    def clear_memory(self):
        self.ptr = 0
        self.total_stored = 0