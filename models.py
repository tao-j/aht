import numpy as np

RAND_CACHE_SIZE = 100000


class Model:
    def __init__(self):
        self.rand_cache = np.random.random(RAND_CACHE_SIZE)
        self.rand_i = 0
        self.rand_i_max = len(self.rand_cache)

        self.Pij = None

    def sample_pair(self, i, j, u=1):
        if self.rand_cache[self.rand_i] < self.Pij[u, i, j]:
            y = 1
        else:
            y = 0
        self.rand_i += 1
        if self.rand_i >= self.rand_i_max:
            self.rand_i = 0
            self.rand_cache = np.random.random(RAND_CACHE_SIZE)
        return y


class DummyModel(Model):
    def __init__(self, a):
        self.original_a = a
        return

    def sample_pair(self, i, j, u=1):
        return 1 if self.original_a[i] > self.original_a[j] else 0


class WSTModel(Model):
    def __init__(self, rank):
        super(WSTModel, self).__init__()
        slackness = 0.01
        self.rank = rank
        self.N = len(rank)
        self.Pij = 0.5 * np.ones([self.N, self.N])
        for i in range(self.N):
            for j in range(i + 1, self.N):
                pij = np.random.random_sample() * (0.5 - slackness) + 0.5 + slackness
                self.Pij[rank[i], rank[j]] = pij
                self.Pij[rank[j], rank[i]] = 1 - pij


class SSTModel(Model):
    def __init__(self, s, gamma=None):
        super().__init__()

        if gamma is None:
            gamma = [1.0]
        self.s = s
        self.gamma = gamma
        self.N = len(s)
        self.M = len(gamma)
        self.Pij = 0.5 * np.ones([self.M, self.N, self.N])

        for u in range(self.M):
            for j in range(self.N):
                for i in range(self.N):
                    self.Pij[u, i, j] = self.pij_func(i, j, u)

    def pij_func(self, i, j, u=1):
        return 0.5


class HBTL(SSTModel):
    def pij_func(self, i, j, u=1):
        return 1. / (1 + np.exp(self.gamma[u] * (self.s[j] - self.s[i])))


class Uniform(SSTModel):
    def pij_func(self, i, j, u=1):
        if self.s[i] > self.s[j]:
            si = 4
            sj = 1
        else:
            si = 1
            sj = 4
        return 1. / (1 + np.exp(self.gamma[u] * (sj - si)))
    # 2.5 0.99
    # 1.0 0.9
    # 0.5 0.81
    # 0.25 0.6
