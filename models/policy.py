import torch.nn as nn
from models.cnn import CNN

class Policy(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.encoder = CNN()
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        features = self.encoder(x)
        return self.actor(features), self.critic(features)