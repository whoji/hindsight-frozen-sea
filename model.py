import torch
import torch.nn as nn

HID_SIZE = 32

class PolicyNet(nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)
