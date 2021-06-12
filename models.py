import numpy as np

RAND_CACHE_SIZE = 100000


class Model:
    def __init__(self, s, gamma):
        self.s = s
        self.gamma = gamma
        self.N = len(s)
        self.M = len(gamma)
        self.Pij = 0.5 * np.ones([self.M, self.N, self.N])

        self.rand_cache = np.random.random(RAND_CACHE_SIZE)
        self.rand_i = 0
        self.rand_i_max = len(self.rand_cache)

        for u in range(self.M):
            for j in range(self.N):
                for i in range(self.N):
                    self.Pij[u, i, j] = self.pij_func(u, i, j)

    def pij_func(self, u, i, j):
        return 0.5

    def sample_pair(self, u, i, j):
        if self.rand_cache[self.rand_i] < self.Pij[u, i, j]:
            y = 1
        else:
            y = 0
        self.rand_i += 1
        if self.rand_i >= self.rand_i_max:
            self.rand_i = 0
            self.rand_cache = np.random.random(RAND_CACHE_SIZE)
        return y


class HBTL(Model):
    def pij_func(self, u, i, j):
        return 1. / (1 + np.exp(self.gamma[u] * (self.s[j] - self.s[i])))


class Uniform(Model):
    def pij_func(self, u, i, j):
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