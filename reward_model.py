import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, input_size):
        super(RewardModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)