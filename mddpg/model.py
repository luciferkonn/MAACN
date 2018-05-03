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
        self.cnn1 = nn.Conv2d(4, 4, 2)
        self.cnn2 = nn.Conv2d(4, 4, 2)
        self.cnn3 = nn.Conv2d(4, 4, 2)
        self.fc1 = nn.Linear(4*3*3, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = F.relu(self.cnn1(x))
        out = F.relu(self.cnn2(out))
        out = F.relu(self.cnn3(out))
        out = out.view(-1, self.num_flat_features(out))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Critic(nn.Module):
    def __init__(self, n_obs, n_actions, hidden1=300, hidden2=600):
        super(Critic, self).__init__()
        self.cnn1 = nn.Conv2d(4, 4, 2)
        self.cnn2 = nn.Conv2d(4, 4, 2)
        self.cnn3 = nn.Conv2d(4, 4, 2)
        self.fc1 = nn.Linear(4 * 3 * 3 + n_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, xs):
        x, a = xs
        out = F.relu(self.cnn1(x))
        out = F.relu(self.cnn2(out))
        out = F.relu(self.cnn3(out))
        out = out.view(-1, self.num_flat_features(out))
        out = self.fc1(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features