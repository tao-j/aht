import random
import numpy as np
from math import ceil, log2, sqrt, log
from models import WSTModel


def trans_closure(T, e):
    # update the transitive closure matrix T
    # T[i][j] = 1 <=> i>j, T[i][j] = -1 <=> i<j
    # T[i][j] = 0 <=> i>j or i<j not determined
    # e=[i,j], to update the relation i<j
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
        alpha_t = sqrt(log((3.14 ** 2 / 3) * (t ** 2) / delta) / (2 * n))
        if phat - 1 / 2 > alpha_t:
            return 1
        elif phat - 1 / 2 < -alpha_t:
            return -1
    return 0


def findmaxmin(rank_slots, T):
    n = len(T)
    L = set()
    U = set()
    ranked_items = set(rank_slots)
    for x in range(n):
        in_L = 1
        in_U = 1
        if x in ranked_items:
            in_L = 0
            in_U = 0
        else:
            for y in range(n):
                if y not in ranked_items:
                    if T[x][y] == -1:
                        in_L = 0
                    if T[x][y] == 1:
                        in_U = 0
        if in_L:
            L.add(x)
        if in_U:
            U.add(x)
    return L, U


class ProbeSort:
    # rank high to low
    def __init__(self, N, delta):
        self.N = N
        self.delta = delta

    def sort(self, a, model):
        n = self.N
        delta = self.delta
        P = model.Pij

        n_comp = np.zeros(n)  # number of comparisons asked involving each item
        rank = np.ones(n) * -1
        T = np.zeros((n, n))
        Cc = np.zeros((n, n))
        Cw = np.zeros((n, n))

        for t in range(n // 2):
            # print('t= =========================', t)
            L, U = findmaxmin(rank, T)
            # print(L, U)
            if len(U) == 1:
                imin = U.pop()
            if len(L) == 1:
                imax = L.pop()
            while len(L) > 0 or len(U) > 0:
                change = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if T[i][j] == 0 and (i in L.union(U)
                                             or j in L.union(U)):
                            y = random.random() < P[i][j]  # ask about i and j once
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
                    if i in U:
                        U.remove(i)
                    if j in L:
                        L.remove(j)
                    # print(L, U)
                    trans_closure(T, [i, j])
                    if len(U) == 1:
                        imin = U.pop()
                    if len(L) == 1:
                        imax = L.pop()
            rank[t] = imax
            rank[n - 1 - t] = imin
            # print(rank)
        if n % 2 == 1:
            rank[n // 2] = int(n * (n - 1) / 2 - sum(rank) - 1)

        # %%
        estnum = [0] * n
        for i, x in enumerate(gt_rank):
            if i == 0:
                estnum[x] += 1 / (P[x][gt_rank[i + 1]] - 1 / 2) ** 2
            elif i == n - 1:
                estnum[x] += 1 / (P[gt_rank[i - 1]][x] - 1 / 2) ** 2
            else:
                estnum[x] += 1 / (P[gt_rank[i - 1]][x] -
                                  1 / 2) ** 2 + 1 / (P[x][gt_rank[i + 1]] - 1 / 2) ** 2
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

        return rank


if __name__ == "__main__":
    random.seed(222)
    np.random.seed(222)
    n = 10
    delta = 0.1

    gt_rank = list(np.random.permutation(n))

    prb_s = ProbeSort(n, delta)
    model = WSTModel(gt_rank)
    rank = prb_s.sort(None, model)

    # print('true ranking:', gt_rank)
    # print('output:', rank)

    assert (np.alltrue(rank == gt_rank))
