import torch
import torch.nn as nn


class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor = nn.Linear(hidden_dim, action_dim)

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        features = self.shared(x)

        logits = self.actor(features)

        value = self.critic(features)

        return logits, value