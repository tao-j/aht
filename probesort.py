from sort import Sort

import random
import numpy as np
from math import ceil, log2, sqrt, log
from models import WSTModel


def trans_closure(T, e):
    # update the transitive closure matrix T
    # T[i][j] = 1 <=> i>j, T[i][j] = -1 <=> i<j
    # T[i][j] = 0 <=> i>j or i<j not determined
    # e=[i,j], to update the relation i>j
    i, j = e
    assert T[i][j] != -1, 'contradiction, opposite order'
    if T[i][j] == 0:
        T[i][j] = 1
        T[j][i] = -1
        for end in range(len(T)):
            if T[j][end] == 1:
                T[i][end] = 1
                T[end][i] = -1
        for start in range(len(T)):
            if T[start][i] == 1:
                for end in range(0, len(T)):
                    if T[i][end] == 1:
                        T[start][end] = 1
                        T[end][start] = -1


def SE(Q, delta):
    # simulate successive elimination
    n = Q[0]
    phat = Q[1] / Q[0]
    if n > 1 and log2(n) == int(log2(n)):
        t = log2(n)
        alpha_t = sqrt(log((np.pi ** 2 / 3) * (t ** 2) / delta) / (2 * n))
        if phat - 1 / 2 > alpha_t:
            return 1
        elif phat - 1 / 2 < -alpha_t:
            return -1
    return 0


def findmaxmin(rank_slots, T):
    n = len(T)
    U = set()
    L = set()
    ranked_items = set(rank_slots)
    for x in range(n):
        in_U = 1
        in_L = 1
        if x in ranked_items:
            in_U = 0
            in_L = 0
        else:
            for y in range(n):
                if y not in ranked_items:
                    if T[x][y] == -1:  # x < y
                        in_U = 0
                    if T[x][y] == 1:  # x > y
                        in_L = 0
        if in_U:
            U.add(x)
        if in_L:
            L.add(x)
    # TODO: should only return 1 item, assert?
    return U, L


class ProbeSort(Sort):
    # rank low to high
    def __init__(self, N, delta, model):
        self.N = N
        self.delta = delta
        self.model = model

        self.sample_complexity = 0

    def arg_sort(self):
        n = self.N
        delta = self.delta

        n_comp = np.zeros(n)  # number of comparisons asked involving each item
        arg_list = n * [-1]
        T = np.zeros((n, n))
        Cc = np.zeros((n, n))
        Cw = np.zeros((n, n))

        for t in range(n // 2):
            # print('t= =========================', t)
            U, L = findmaxmin(arg_list, T)
            # print(U, L)
            if len(L) == 1:
                imin = L.pop()
            if len(U) == 1:
                imax = U.pop()
            while len(U) > 0 or len(L) > 0:
                change = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if T[i][j] == 0 and (i in U.union(L)
                                             or j in U.union(L)):
                            y = self.model.sample_pair(i, j)
                            n_comp[i] += 1
                            n_comp[j] += 1
                            Cc[i, j] += 1
                            Cw[i, j] += y
                            if SE([Cc[i, j], Cw[i, j]], 2 * delta / n / n) == 1:  # means i>j
                                # print(i, '>', j)
                                change.append([i, j])
                            elif SE([Cc[i, j], Cw[i, j]],
                                    2 * delta / n ** 2) == -1:  # means i<j
                                # print(i, '<', j)
                                change.append([j, i])
                for i, j in change:
                    # print(i, j)
                    if i in L:
                        L.remove(i)
                    if j in U:
                        U.remove(j)
                    # print(U, L)
                    trans_closure(T, [i, j])
                    if len(L) == 1:
                        imin = L.pop()
                    if len(U) == 1:
                        imax = U.pop()
            arg_list[t] = imax
            arg_list[n - 1 - t] = imin
            # print(arg_list)
        if n % 2 == 1:
            arg_list[n // 2] = int(n * (n - 1) / 2 - sum(arg_list) - 1)

        # %%
        # estnum = [0] * n
        # for i, x in enumerate(gt_rank):
        #     if i == 0:
        #         estnum[x] += 1 / (P[x][gt_rank[i + 1]] - 1 / 2) ** 2
        #     elif i == n - 1:
        #         estnum[x] += 1 / (P[gt_rank[i - 1]][x] - 1 / 2) ** 2
        #     else:
        #         estnum[x] += 1 / (P[gt_rank[i - 1]][x] -
        #                           1 / 2) ** 2 + 1 / (P[x][gt_rank[i + 1]] - 1 / 2) ** 2
        # print(n_comp)
        # print(estnum)
        self.sample_complexity = np.sum(n_comp)

        # from matplotlib import pyplot as plt
        #
        # # fig, axs = plt.subplots(2)
        # fig1 = plt.figure()
        # plt.plot(estnum)
        # fig2 = plt.figure()
        # plt.plot(n_comp)

        return list(reversed(arg_list))


class ProbeSortA(ProbeSort):
    def SC(self, i, j, delta, tau):
        w_i = 0
        eps_tau = pow(2, -tau)
        delta_tau = delta / np.pi / np.pi * 6 / tau / tau
        b_tau = 2 / eps_tau / eps_tau * log(1. / delta_tau)
        b_tau = int(ceil(b_tau))
        for t in range(b_tau):
            y = self.model.sample_pair(i, j)
            w_i += y
        phat = w_i / b_tau

        ans = 0
        if phat - 1 / 2 > 0.5 * eps_tau:
            ans = 1
        elif phat - 1 / 2 < -0.5 * eps_tau:
            ans = -1
        return ans, b_tau

    def arg_sort(self):
        n = self.N
        delta = self.delta

        n_comp = np.zeros(n)  # number of comparisons asked involving each item
        arg_list = n * [-1]
        T = np.zeros((n, n))
        Tau = np.ones((n, n))

        for t in range(n - 1):
            # print('t= =========================', t)
            U, _ = findmaxmin(arg_list, T)
            # print(U)
            if len(U) == 1:
                imax = U.pop()
            while len(U) > 0:
                change = []
                Tau_now = np.array(Tau)
                for i in range(n):
                    if i not in U:
                        Tau_now[i, :] = float("inf")
                i, j = np.unravel_index(Tau_now.argmin(), Tau_now.shape)
                assert i in U or j in U
                tau_ij = Tau[i][j]
                Tau[i][j] += 1
                Tau[j][i] += 1
                ans, cost = self.SC(i, j, 2 * delta / n / n, tau_ij)
                n_comp[i] += cost
                n_comp[j] += cost
                if ans == 1:
                    # print(i, '>', j)
                    change.append([i, j])
                elif ans == -1:
                    # print(i, '<', j)
                    change.append([j, i])
                for i, j in change:
                    # print(i, j)
                    if j in U:
                        U.remove(j)
                    # print(U, "in change")
                    trans_closure(T, [i, j])
                    if len(U) == 1:
                        imax = U.pop()
            Tau[imax, :] = float("inf")
            Tau[:, imax] = float("inf")
            arg_list[t] = imax
            # print(arg_list)
        arg_list[n - 1] = findmaxmin(arg_list, T)[0].pop()
        self.sample_complexity = np.sum(n_comp)
        return list(reversed(arg_list))


class ProbeSortB(ProbeSortA):
    def arg_sort(self):
        n = self.N
        delta = self.delta

        n_comp = np.zeros(n)  # number of comparisons asked involving each item
        arg_list = n * [-1]
        T = np.zeros((n, n))
        Tau = np.ones((n, n))

        for t in range(n // 2):
            # print('t= =========================', t)
            U, L = findmaxmin(arg_list, T)
            # print(U)
            if len(U) == 1:
                imax = U.pop()
            if len(L) == 1:
                imin = L.pop()
            while len(U) > 0 or len(L) > 0:
                change = []
                Tau_now = np.array(Tau)
                for i in range(n):
                    if i not in U and i not in L:
                        Tau_now[i, :] = float("inf")
                i, j = np.unravel_index(Tau_now.argmin(), Tau_now.shape)
                assert i in U or j in U or i in L or j in L
                tau_ij = Tau[i][j]
                Tau[i][j] += 1
                Tau[j][i] += 1
                ans, cost = self.SC(i, j, 2 * delta / n / n, tau_ij)
                n_comp[i] += cost
                n_comp[j] += cost
                if ans == 1:
                    # print(i, '>', j)
                    change.append([i, j])
                elif ans == -1:
                    # print(i, '<', j)
                    change.append([j, i])
                for i, j in change:
                    # print(i, j)
                    if i in L:
                        L.remove(i)
                    if j in U:
                        U.remove(j)
                    # print(U, "in change")
                    trans_closure(T, [i, j])
                    if len(L) == 1:
                        imin = L.pop()
                    if len(U) == 1:
                        imax = U.pop()
            Tau[imax, :] = float("inf")
            Tau[:, imax] = float("inf")
            Tau[imin, :] = float("inf")
            Tau[:, imin] = float("inf")
            arg_list[t] = imax
            arg_list[n - 1 - t] = imin
            # print(arg_list)
        if n % 2 == 1:
            arg_list[n // 2] = int(n * (n - 1) / 2 - sum(arg_list) - 1)

        # print(arg_list)
        self.sample_complexity = np.sum(n_comp)
        return list(reversed(arg_list))

class ProbeSortOB(ProbeSort):
    def arg_sort(self):
        n = self.N
        delta = self.delta

        n_comp = np.zeros(n)  # number of comparisons asked involving each item
        arg_list = n * [-1]
        T = np.zeros((n, n))
        Cc = np.zeros((n, n))
        Cw = np.zeros((n, n))

        for t in range(n - 1):
            print('t= =========================', t)
            U, _ = findmaxmin(arg_list, T)
            print(U, " ")
            if len(U) == 1:
                imax = U.pop()
            while len(U) > 0:
                change = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if T[i][j] == 0 and (i in U):
                            y = self.model.sample_pair(i, j)
                            n_comp[i] += 1
                            n_comp[j] += 1
                            Cc[i, j] += 1
                            Cw[i, j] += y
                            if SE([Cc[i, j], Cw[i, j]], 2 * delta / n / n) == 1:  # means i>j
                                # print(i, '>', j)
                                change.append([i, j])
                            elif SE([Cc[i, j], Cw[i, j]],
                                    2 * delta / n ** 2) == -1:  # means i<j
                                # print(i, '<', j)
                                change.append([j, i])
                for i, j in change:
                    # print(i, j)
                    if j in U:
                        U.remove(j)
                    # print(U, L)
                    trans_closure(T, [i, j])
                    if len(U) == 1:
                        imax = U.pop()
            arg_list[t] = imax
            # print(arg_list)
            # arg_list[n - 1] = findmaxmin(arg_list, T)[0].pop()

        self.sample_complexity = np.sum(n_comp)

        return list(reversed(arg_list))

if __name__ == "__main__":
    random.seed(222)
    np.random.seed(222)
    n = 10
    delta = 0.01

    gt_rank = list(np.random.permutation(n))

    wst_m = WSTModel(gt_rank)
    prb_s = ProbeSort(n, delta, wst_m)
    prb_a = prb_s.arg_sort()

    prba_s = ProbeSortA(n, delta, wst_m)
    prba_a = prba_s.arg_sort()

    prbb_s = ProbeSortOB(n, delta, wst_m)
    prbb_a = prbb_s.arg_sort()


    print('true ranking:', gt_rank)
    print('output o:', prb_a, prb_s.sample_complexity)
    print('output a:', prba_a, prba_s.sample_complexity)
    print('output b:', prbb_a, prbb_s.sample_complexity)

    assert (np.alltrue(prb_a == gt_rank))
    assert (np.alltrue(prba_a == gt_rank))
t