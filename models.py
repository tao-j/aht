import numpy as np

RAND_CACHE_SIZE = 100000


class Model:
    def __init__(self):
        self.rand_cache = np.random.random(RAND_CACHE_SIZE)
        self.rand_i = 0
        self.rand_i_max = len(self.rand_cache)

        self.Pij = None

    def sample_pair(self, i, j, u=0):
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
        self.array_original = a
        return

    def sample_pair(self, i, j, u=0):
        return 1 if self.array_original[i] > self.array_original[j] else 0


class WSTModel(Model):
    def __init__(self, rank, delta_d=0.25):
        super(WSTModel, self).__init__()
        self.rank = rank
        self.N = len(rank)
        self.M = 1
        self.Pij = 0.5 * np.ones([self.M, self.N, self.N])
        assert delta_d < 0.5
        self.init_matrix(rank, delta_d)

    def init_matrix(self, rank, delta_d):
        for i in range(self.N):
            for j in range(i + 1, self.N):
                pij = np.random.random_sample() * (0.5 - delta_d) + 0.5 + delta_d
                self.Pij[0, rank[i], rank[j]] = 1 - pij
                self.Pij[0, rank[j], rank[i]] = pij


class AdjacentOnlyModel(WSTModel):
    def init_matrix(self, rank, delta_d):
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # adj 0.5+d/2
                # non-adj 0.5+sqrt(1/n)
                if j == i + 1:
                    pij = 0.50 + delta_d
                else:
                    pij = 0.5 + (1 / self.N)
                self.Pij[0, rank[i], rank[j]] = 1 - pij
                self.Pij[0, rank[j], rank[i]] = pij


class WSTAdjModel(WSTModel):
    def init_matrix(self, rank, delta_d):
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # adj 0.5+d/10+d~1
                # non adj 0.5+d/10~0.5+d/10+d
                min_p = delta_d / 10 + 0.5
                if j == i + 1:
                    pij = np.random.random_sample() * (1 - min_p - delta_d) + min_p + delta_d
                else:
                    pij = np.random.random_sample() * delta_d + min_p
                self.Pij[0, rank[i], rank[j]] = 1 - pij
                self.Pij[0, rank[j], rank[i]] = pij


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

    def pij_func(self, i, j, u=0):
        raise NotImplementedError


class HBTL(SSTModel):
    def pij_func(self, i, j, u=0):
        return 1. / (1 + np.exp(self.gamma[u] * (self.s[j] - self.s[i])))


class Uniform(SSTModel):
    def pij_func(self, i, j, u=0):
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
