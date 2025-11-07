import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, num_generators):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, num_generators * 2)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        x = self.network(state)
        return torch.clamp(x, min=-10, max=10)


class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state):
        return self.network(state)


class ImprovedDNNModel(nn.Module):
    def __init__(self, input_dim, num_buses=30):
        super().__init__()
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.convergence_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.voltage_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_buses)
        )

        self.angle_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_buses)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.shared_features(x)
        convergence_logits = self.convergence_predictor(features)
        vm = self.voltage_predictor(features)
        va = self.angle_predictor(features)
        return convergence_logits, vm, va