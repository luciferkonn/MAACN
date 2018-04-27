import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# def fanin_init(size, fanin=None):
#     fanin = fanin or size[0]
#     v = 1. / np.sqrt(fanin)
#     return torch.Tensor

class Actor(nn.Module):
    def __init__(self, n_obs, n_actions, hidden1=300, hidden2=600):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_obs, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, n_obs, n_actions, hidden1=300, hidden2=600):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_obs, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
