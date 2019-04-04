import numpy as np
import random
import torch

class ReplayBuffer(object):

    def __init__(self, size):

        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, R, preact, var, d):
        data = (obs_t, action, R, preact, var, d)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, returns, preacts, vars, ds = [], [], [], [] ,[], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, R, preact, var, d = data
            obses_t.append(obs_t)
            actions.append(action)
            returns.append(R)
            preacts.append(preact)
            vars.append(var)
            ds.append(d)

        return torch.stack(obses_t), torch.stack(actions), torch.stack(returns), torch.stack(preacts), torch.stack(vars), torch.stack(ds)

    def sample(self, batch_size):

        # print(batch_size)
        # print(len(self._storage))
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    def get_buffersize(self):
        return len(self._storage)


