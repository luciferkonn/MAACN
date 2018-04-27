import random
import numpy as np

class ReplayMemory(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs, action, reward, obs_next, done):
        data = (obs, action, reward, obs_next, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses, actions, rewards, obses_next, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs, action, reward, obs_next, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_next.append(np.array(obses_next, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_next), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample()

    def collect(self):
        return self.sample(-1)
            


