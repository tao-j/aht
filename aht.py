import numpy as np
import random
from cmpsort import CmpSort


class ActiveRank:
    def __init__(self, N, M, delta, s, gamma, active=True):
        self.N = N
        self.M = M
        self.cU = np.array(range(0, M))
        self.s = s
        self.gamma = gamma
        self.delta = delta
        self.cmp_sort = CmpSort(s, delta)

        self.rank_sample_complexity = 0
        self.active = active

    def eliminate_user(self, eps=0.1, delta=0.1):
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
                pack_a = self.atc(pair[0], pair[1], self.cmp_sort.epsilon_atc_param, self.cmp_sort.delta_atc_param,
                                  self.cmp_sort.ranked_list, self.s, self.gamma)
                pack_b = self.cmp_sort.feedback(pack_a[0])
                if self.active:
                    self.post_atc(pack_a, pack_b)

        return self.rank_sample_complexity, self.cmp_sort.ranked_list

    def atc(self, i, j, eps, delta, ranked_s, original_s, gamma):
        pass

    def post_atc(self, pack_a, pack_b):
        pass

    def init_user_counter(self):
        raise NotImplementedError

    def update_user_counter(self):
        raise NotImplementedError


class TwoStageSimultaneousActiveRank(ActiveRank):
    def __init__(self, N, M, delta, s, gamma, active=True):
        super().__init__(N, M, delta, s, gamma, active)
        self.n_t = np.zeros(M)
        self.s_t = 0

    def post_atc(self, pack_a, pack_b):
        y, bn, r = pack_a
        self.s_t += r
        self.n_t += bn
        self.cU = self.eliminate_user(delta=delta)

    def eliminate_user(self, eps=0.1, delta=0.1):
        if len(self.cU) == 1:
            return self.cU
        s_max = int(np.ceil(2 / eps / eps * np.log2(len(self.cU) / delta)))
        if self.s_t > s_max:
            mu_t = self.n_t / self.s_t
            i_best = np.argmax(mu_t)
            self.cU = [i_best]
        return self.cU

    def atc(self, i, j, eps, delta, ranked_s, original_s, gamma):
        """
        Do AttemptToCompare in rounds. One round asks every user once.
        """
        w = 0
        m_t = len(self.cU)
        b_max = np.ceil(1. / 2 / m_t / eps ** 2 * np.log(2 / delta))
        bn = np.zeros(self.M)
        p = 0.5
        r = 0
        for t in range(1, int(b_max)):
            for u in self.cU:
                # s_i = original_s[i] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
                # s_j = ranked_s[j] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
                pij = np.exp(gamma[u] * original_s[i]) / (
                        np.exp(gamma[u] * original_s[i]) + np.exp(gamma[u] * ranked_s[j]))
                # if original_s[i] > ranked_s[j]:
                #     pij = np.exp(gamma[u] * 4) / (
                #             np.exp(gamma[u] * 4) + np.exp(gamma[u] * 1))
                # else:
                #     pij = np.exp(gamma[u] * 1) / (
                #             np.exp(gamma[u] * 4) + np.exp(gamma[u] * 1))
                y = 1 if np.random.random() < pij else 0
                if y == 1:
                    w += 1
                    bn[u] += 1
            r = t
            b_t = np.sqrt(1. / 2 / (r + 1) / m_t * np.log(np.pi ** 2 * (r + 1) ** 2 / 3 / delta))
            p = w / r / len(self.cU)
            if p > 0.5 + b_t:
                break
            if p < 0.5 - b_t:
                break

        atc_y = 1 if p > 0.5 else 0
        bn = bn if p > 0.5 else r - bn
        self.rank_sample_complexity += r * len(self.cU)
        return atc_y, bn, r


class TwoStageSeparateRank(TwoStageSimultaneousActiveRank):
    def __init__(self, N, M, delta, s, gamma, active=True):
        super().__init__(N, M, delta, s, gamma, active)
        # rank the first pair of item
        algo = TwoStageSimultaneousActiveRank(2, M, delta, s[:2], gamma, active=False)
        cost1, ranked = algo.rank()
        if ranked[0] != s[0]:
            self.gt_y = 1
        else:
            self.gt_y = 0
        eps_user, cost2 = self.eliminate_user()
        self.rank_sample_complexity += cost2 + cost1

    def post_atc(self, pack_a, pack_b):
        pass

    def eliminate_user(self, eps=0.50, delta=0.25):
        # medium elimination
        eps = eps / 4
        delta = delta / 2

        bn = np.zeros(self.M)
        bs = np.zeros(self.M)
        while True:
            if len(self.cU) == 1:
                return self.cU, np.sum(bs)
            b_max = int(np.ceil(4 / eps / eps * np.log2(3 / delta)))
            for t in range(1, int(b_max)):
                for u in self.cU:
                    bs[u] += 1
                    # s_i = original_s[i] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
                    # s_j = ranked_s[j] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
                    # pij = np.exp(self.gamma[u] * self.s[i]) / (
                            # np.exp(self.gamma[u] * self.s[i]) + np.exp(self.gamma[u] * self.ranked_s[j]))
                    if self.s[0] > self.s[1]:
                        pij = np.exp(self.gamma[u] * 4) / (
                                np.exp(self.gamma[u] * 4) + np.exp(self.gamma[u] * 1))
                    else:
                        pij = np.exp(self.gamma[u] * 1) / (
                                np.exp(self.gamma[u] * 4) + np.exp(self.gamma[u] * 1))
                    y = 1 if np.random.random() < pij else 0
                    if y == 1:
                        bn[u] += 1
            eps = 3 / 4 * eps
            delta = delta / 2
            mu = bn / bs
            if self.gt_y == 0:
                mu = 1 - mu
            ranked_u_cm = np.sort(mu[self.cU])
            ranked_u_idx = np.argsort(mu[self.cU])
            keep = len(self.cU) // 2
            self.cU = self.cU[ranked_u_idx[keep:]]

    def rank(self):
        cost, ranked = super().rank()
        return cost, ranked


class UnevenUCBActiveRank(ActiveRank):
    def __init__(self, N, M, delta, s, gamma, active=True):
        super().__init__(N, M, delta, s, gamma, active)
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
            inserted_idx = len(self.cmp_sort.ranked_list)
            for j in range(inserted_idx):
                if inserted_place > j:
                    self.bn += self.A[j]
                elif inserted_place < j:
                    self.bn += self.B[j - 1]
            # assert (np.sum(self.A, axis=0) + np.sum(self.B, axis=0) + self.bn == self.bs).all()
            self.A, self.B = self.create_mat(self.N, self.M)
            self.eliminate_user()

    def atc(self, i, j, eps, delta, ranked_s, original_s, gamma):
        t_max = int(np.ceil(1. / 2 / (eps ** 2) * np.log(2 / delta)))
        p = 0.5
        w = 0
        for t in range(1, t_max + 1):
            u = np.random.choice(self.cU, 1)[0]
            self.bs[u] += 1
            self.rank_sample_complexity += 1
            # s_i = original_s[i] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
            # s_j = ranked_s[j] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
            pij = np.exp(gamma[u] * original_s[i]) / (
                    np.exp(gamma[u] * original_s[i]) + np.exp(gamma[u] * ranked_s[j]))
            # assert original_s[i] != ranked_s[j]
            # if original_s[i] > ranked_s[j]:
            #     pij = np.exp(gamma[u] * 4) / (
            #         np.exp(gamma[u] * 4) + np.exp(gamma[u] * 1))
            # else:
            #     pij = np.exp(gamma[u] * 1) / (
            #         np.exp(gamma[u] * 4) + np.exp(gamma[u] * 1))
            y = 1 if np.random.random() < pij else 0 # i > j
            if y == 1:
                self.A[j, u] += 1
                w += 1
            else:
                self.B[j, u] += 1
            b_t = np.sqrt(1. / 2 / t * np.log(np.pi * np.pi * t * t / 3 / delta))
            p = w / t
            if p > 0.5 + b_t:
                break
            if p < 0.5 - b_t:
                break

        atc_y = 1 if p > 0.5 else 0
        return atc_y, self.A, self.bs

    def eliminate_user(self, eps=0.1, delta=0.1):
        smin = min(self.bs[self.cU])
        mu = self.bn / self.bs
        if smin == 0:
            assert False
        if np.log2(2 * len(self.cU) / delta) / 2 / smin < 0:
            assert False
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
