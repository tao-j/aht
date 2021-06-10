import numpy as np
import random
from pitsort import PITSort
from models import HBTL, Uniform

RAND_CACHE_SIZE = 100000


class ActiveRank:
    def __init__(self, N, M, eps_user, delta_rank, delta_user, s, gamma, active=True):
        self.N = N
        self.M = M
        self.cU = np.array(range(0, M))
        self.s = s
        self.gamma = gamma
        self.eps_user = eps_user
        self.delta_rank = delta_rank
        self.delta_user = delta_user

        self.cmp_sort = PITSort(N, delta_rank)
        self.model = Uniform(s, gamma)
        # self.model = HBTL(s, gamma)

        self.rank_sample_complexity = 0
        self.active = active

        self.rand_i_max = RAND_CACHE_SIZE
        self.rand_i = np.zeros(self.M, dtype=np.int)
        self.mt = len(self.cU)
        self.rand_cache = np.zeros([self.M, RAND_CACHE_SIZE], dtype=np.int)
        for ui in range(0, self.M):
            self.rand_cache[ui] = np.random.randint(0, ui + 1, RAND_CACHE_SIZE)

    def sample_user_idx(self):
        self.mt = len(self.cU) - 1
        assert self.mt >= 0
        u = self.rand_cache[self.mt, self.rand_i[self.mt]]
        self.rand_i[self.mt] += 1
        if self.rand_i[self.mt] >= self.rand_i_max:
            self.rand_i[self.mt] = 0
            self.rand_cache[self.mt] = np.random.randint(0, self.mt + 1, RAND_CACHE_SIZE)

        return u

    def eliminate_user(self):
        pass

    def rank(self):
        while not self.cmp_sort.done:
            pair = self.cmp_sort.next_pair()
            assert (0 <= pair[0] <= self.cmp_sort.n_intree)
            assert (-1 <= pair[1] <= self.cmp_sort.n_intree)
            if pair[1] == -1:
                self.cmp_sort.feedback(1)
            elif pair[1] == self.cmp_sort.n_intree:
                self.cmp_sort.feedback(0)
            else:
                pack_a = self.atc(pair[0], self.cmp_sort.arg_list[pair[1]], self.cmp_sort.epsilon_atc_param, self.cmp_sort.delta_atc_param,
                                  self.cmp_sort.arg_list, self.s, self.gamma)
                pack_b = self.cmp_sort.feedback(pack_a[0])
                if self.active:
                    self.post_atc(pack_a, pack_b)

        return self.rank_sample_complexity, self.cmp_sort.arg_list

    def atc(self, i, j, eps, delta, ranked_s, original_s, gamma):
        pass

    def post_atc(self, pack_a, pack_b):
        pass

    # def init_user_counter(self):
    #     raise NotImplementedError
    #
    # def update_user_counter(self):
    #     raise NotImplementedError


class UnevenUCBActiveRank(ActiveRank):
    def __init__(self, N, M, eps_user, delta_rank, delta_user, s, gamma, active=True):
        super().__init__(N, M, eps_user, delta_rank, delta_user, s, gamma, active)
        # number of times user is asked
        self.bs = np.zeros(M)
        # number of times user is correct
        self.bn = np.zeros(M)
        # temp matrix list holding user response
        self.A, self.B = self.create_mat(N, M)

    @staticmethod
    def create_mat(N, M):
        return np.zeros((N, M)), np.zeros((N, M))

    def post_atc(self, pack_a, pack_b):
        inserted, inserted_place = pack_b
        if inserted:
            assert inserted_place != -1
            inserted_idx = len(self.cmp_sort.arg_list)
            for j in range(inserted_idx):
                if inserted_place > j:
                    self.bn += self.A[j]
                elif inserted_place < j:
                    self.bn += self.B[j - 1]
            # assert (np.sum(self.A, axis=0) + np.sum(self.B, axis=0) + self.bn == self.bs).all()
            self.A, self.B = self.create_mat(self.N, self.M)
            self.eliminate_user()

    def atc(self, i, j, eps, delta, ranked_s, original_s, gamma):
        t_max = int(np.ceil(1. / 2 / (eps ** 2) * np.log2(2 / delta)))
        p = 0.5
        w = 0
        t = np.arange(1, t_max + 1)
        bb_t = np.sqrt(1. / 2 / t * np.log2(np.pi * np.pi * t * t / 3 / delta))
        for t in range(1, t_max + 1):
            u = self.cU[self.sample_user_idx()]
            self.bs[u] += 1
            y = self.model.sample_pair(u, i, j)
            if y == 1:
                self.A[j, u] += 1
                w += 1
            else:
                self.B[j, u] += 1
            b_t = bb_t[t - 1]
            p = w / t
            if p > 0.5 + b_t:
                break
            if p < 0.5 - b_t:
                break

        self.rank_sample_complexity += w
        atc_y = 1 if p > 0.5 else 0
        return atc_y, self.A, self.bs

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
            self.cU = new_cM

        return r


class TwoStageSeparateRank(UnevenUCBActiveRank):
    def __init__(self, N, M, eps_user, delta_rank, delta_user, s, gamma, active=True):
        super().__init__(N, M, eps_user, delta_rank, delta_user, s, gamma, active)
        # rank the first pair of item
        algo = UnevenUCBActiveRank(2, M, eps_user, delta_rank, delta_user, s[:2], gamma, active=False)
        cost1, ranked = algo.rank()
        if ranked[0] != s[0]:
            self.gt_y = 1
        else:
            self.gt_y = 0
        r = self.eps_user
        while r >= self.eps_user:
            u = self.cU[self.sample_user_idx()]
            y = self.model.sample_pair(u, 0, 1)
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

    def rank(self):
        cost, ranked = super().rank()
        return cost, ranked
