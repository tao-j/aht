import numpy as np

RAND_CACHE_SIZE = 100000


def rnd_rng(a, b):
    return np.random.random_sample() * (b - a) + a


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
                    pij = 0.5 + delta_d * (1 / self.N)
                self.Pij[0, rank[i], rank[j]] = 1 - pij
                self.Pij[0, rank[j], rank[i]] = pij


class AdjacentConstantModel(WSTModel):
    def init_matrix(self, rank, delta_d):
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # adj 0.5+d/2
                # non-adj 0.5+sqrt(1/n)
                if j == i + 1:
                    pij = 0.50 + 0.40
                else:
                    pij = 0.5 + delta_d
                self.Pij[0, rank[i], rank[j]] = 1 - pij
                self.Pij[0, rank[j], rank[i]] = pij


class AdjacentSqrtModel(WSTModel):
    def init_matrix(self, rank, delta_d):
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # adj 0.5+d/2
                # non-adj 0.5+sqrt(1/n)
                if j == i + 1:
                    pij = 0.50 + delta_d
                else:
                    pij = 0.5 + delta_d * np.sqrt(1 / self.N)
                self.Pij[0, rank[i], rank[j]] = 1 - pij
                self.Pij[0, rank[j], rank[i]] = pij


class WSTAdjModel(WSTModel):
    def init_matrix(self, rank, delta_d):
        for i in range(self.N):
            for j in range(i + 1, self.N):
                pij_adj = rnd_rng(0.5 + delta_d, 1)
                pij_njj = rnd_rng(0.5 + delta_d / 10, 0.5 + delta_d)
                if j == i + 1:
                    pij = pij_adj
                else:
                    pij = pij_njj
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


class Scaling:
    def scale_by_gamma(self):
        gamma = self.gamma
        self.Pij = (self.Pij - 0.5) * gamma[:, np.newaxis, np.newaxis] + 0.5
        assert np.all(self.Pij <= 1)
        assert np.all(self.Pij >= 0)


class SSTScale(HBTL, Scaling):
    def __init__(self, s, gamma=np.ones(1)):
        gamma = np.array(gamma)
        super().__init__(s, np.ones(gamma.shape))
        self.gamma = gamma
        self.scale_by_gamma()


class WSTScale(WSTModel, Scaling):
    def __init__(self, rank, delta_d=0.25, gamma=np.ones(1)):
        super().__init__(rank, delta_d=delta_d)
        self.gamma = gamma
        self.scale_by_gamma()
