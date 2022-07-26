import numpy as np
import random
from sort_pit import PITSort
from models import HBTL, Uniform

RAND_CACHE_SIZE = 100000


class HeterogeneousPITSort(PITSort):
    def __init__(self, n, m, eps_user, delta_rank, delta_user, model, active=True):
        self.M = m
        self.cU = np.array(range(0, m))
        self.eps_user = eps_user
        self.delta_rank = delta_rank
        self.delta_user = delta_user
        super(HeterogeneousPITSort, self).__init__(n, delta_rank, model)

        self.rank_sample_complexity = 0
        self.active = active

        self.rand_i_max = RAND_CACHE_SIZE
        self.rand_i = np.zeros(self.M, dtype=np.int)
        self.mt = len(self.cU)
        self.rand_cache = np.zeros([self.M, RAND_CACHE_SIZE], dtype=np.int)
        for ui in range(0, self.M):
            self.rand_cache[ui] = np.random.randint(0, ui + 1, RAND_CACHE_SIZE)

        # number of times user is asked
        self.bs = np.zeros(m)
        # number of times user is correct
        self.bn = np.zeros(m)
        # temp matrix list holding user response
        self.A, self.B = self.create_mat(n, m)

    def sample_user_idx(self):
        self.mt = len(self.cU) - 1
        assert self.mt >= 0
        u = self.rand_cache[self.mt, self.rand_i[self.mt]]
        self.rand_i[self.mt] += 1
        if self.rand_i[self.mt] >= self.rand_i_max:
            self.rand_i[self.mt] = 0
            self.rand_cache[self.mt] = np.random.randint(0, self.mt + 1, RAND_CACHE_SIZE)

        return u

    @staticmethod
    def create_mat(N, M):
        return np.zeros((N, M)), np.zeros((N, M))

    def post_atc(self, inserted, inserted_place):
        if inserted and self.active:
            assert inserted_place != -1
            arg_list = self.arg_list
            arg_list_len = len(arg_list)
            for idx in range(arg_list_len):
                j = arg_list[idx]
                if inserted_place > idx:
                    self.bn += self.A[j]
                elif inserted_place < idx:
                    self.bn += self.B[j]
            # assert (np.sum(self.A, axis=0) + np.sum(self.B, axis=0) + self.bn == self.bs).all()
            self.A, self.B = self.create_mat(self.N, self.M)
            self.eliminate_user()

    def request_pair(self, i, j):
        u = self.cU[self.sample_user_idx()]
        self.bs[u] += 1
        y = self.model.sample_pair(i, j, u)
        if y == 1:
            self.A[j, u] += 1
        else:
            self.B[j, u] += 1
        return y

    def eliminate_user(self):
        smin = min(self.bs[self.cU])
        mu = self.bn / (self.bs + 1e-10)
        eps = self.eps_user
        delta = self.delta_user
        if smin == 0:
            return eps
        assert np.log2(2 * len(self.cU) / delta) / 2 / smin > 0
        r = np.sqrt(np.log2(2 * len(self.cU) / delta) / 2 / smin)
        stotal = sum(self.bs)
        if stotal > 2 * self.M * self.M * np.log2(self.N * self.M / delta):
            bucb = mu + r
            blcb = mu - r
            to_remove = set()
            for u in self.cU:
                for up in self.cU:
                    if bucb[u] < blcb[up]:
                        to_remove.add(u)
                        break
            new_cM = []
            for u in self.cU:
                if u not in to_remove:
                    new_cM.append(u)
            if new_cM == []:
                assert False
            # if set(self.cU) != set(new_cM):
            #     print(self.cmp_sort.n_intree, to_remove)
            self.cU = new_cM

        return r


class TwoStageSeparateRank(HeterogeneousPITSort):
    def __init__(self, n, m, eps_user, delta_rank, delta_user, model, active=True):
        super().__init__(n, m, eps_user, delta_rank, delta_user, model, active)
        # rank the first pair of item
        algo = HeterogeneousPITSort(2, m, eps_user, delta_rank, delta_user, model.__class__(model.s[:2], model.gamma),
                                    active=False)
        ranked = algo.arg_sort()
        cost1 = algo.sample_complexity
        if ranked[0] != 0:
            self.gt_y = 1
        else:
            self.gt_y = 0
        r = self.eps_user
        while r >= self.eps_user:
            u = self.cU[self.sample_user_idx()]
            y = self.model.sample_pair(0, 1, u)
            self.bs[u] += 1
            if y == self.gt_y:
                self.bn[u] += 1
            self.rank_sample_complexity += 1
            r = self.eliminate_user()
        # cost_naive = 4 * np.log2(2 * self.M / delta_rank) / (eps ** 2) * self.M
        # print(f"naive {cost_naive * 64}, medium {cost2}")
        # self.rank_sample_complexity += cost_naive + cost1
        self.rank_sample_complexity += cost1

    def post_atc(self, pack_a, pack_b):
        pass
