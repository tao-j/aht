# %%
import random
import numpy as np
from math import ceil, log2, sqrt, log
from models import WSTModel


def twopower(x):
    return 2 ** ceil(log2(x))


def trans_closure(T, e):
    # update the transitive closure matrix T
    # in T, T[i][j] = 1 <=> i>j, T[i][j] = -1 <=> i<j, T[i][j] = 0 <=> i>j or i<j not determined
    # e=[i,j], to update the relation i<j
    i, j = e
    if T[i][j] == -1:
        print('contradiction, opposite order')
        return
    elif T[i][j] == 0:
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


def adjacent(p):
    # return an adjacency matrix A from permutation p where A[i][j] = 1 means i>j and A[i][j] = -1 means i<j
    n = len(p)
    A = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            x = p[i]
            y = p[j]
            A[x][y] = 1
            A[y][x] = -1
    return A


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


def findmaxmin(rank, T):
    n = len(T)
    L = set()
    M = set()
    for a in range(n):
        ifinL = 1
        ifinM = 1
        if a in rank:
            ifinL = 0
            ifinM = 0
        else:
            for b in range(n):
                if b not in rank:
                    if T[a][b] == -1:
                        ifinL = 0
                    if T[a][b] == 1:
                        ifinM = 0
        if ifinL == 1:
            L.add(a)
        if ifinM == 1:
            M.add(a)
    return L, M


# %%
if __name__ == "__main__":
    # rank high to low

    # random.seed(222)
    # np.random.seed(222)
    n = 9
    gt_rank = list(np.random.permutation(n))

    wm = WSTModel(gt_rank)
    P = wm.Pij

    # start probe sorting
    delta = 0.1
    n_comp = np.zeros(n)  # number of comparisons asked involving each item
    rank = np.ones(n) * -1
    T = np.zeros((n, n))  # initialize the transitive closure
    Q = {}  # initialize the queries Q
    for i in range(n):
        for j in range(i + 1, n):
            Q[tuple([i, j])] = [0, 0]

    for t in range(n // 2):
        print('t=', t)
        L, M = findmaxmin(rank, T)
        print(L, M)
        if len(M) == 1:
            imin = M.pop()
        if len(L) == 1:
            imax = L.pop()
        while len(L) > 0 or len(M) > 0:
            change = []
            for i in range(n):
                for j in range(i + 1, n):
                    if T[i][j] == 0 and i != j and (i in L.union(M)
                                                    or j in L.union(M)):
                        x = random.random() < P[i][j]  # ask about i and j once
                        n_comp[i] += 1
                        n_comp[j] += 1
                        Q[tuple([i, j])][0] += 1
                        Q[tuple([i, j])][1] += x
                        if SE(Q[tuple([i, j])], 2 * delta / n / n) == 1:  # means i>j
                            print(i, '>', j)
                            change.append([i, j])
                        elif SE(Q[tuple([i, j])],
                                2 * delta / n ** 2) == -1:  # means i<j
                            print(i, '<', j)
                            change.append([j, i])
            for i, j in change:
                print(i, j)
                if i in M:
                    M.remove(i)
                if j in L:
                    L.remove(j)
                print(L, M)
                trans_closure(T, [i, j])
                if len(M) == 1:
                    imin = M.pop()
                if len(L) == 1:
                    imax = L.pop()
        rank[t] = imax
        rank[n - 1 - t] = imin
        print(rank)
    if n % 2 == 1:
        rank[n // 2] = int(n * (n - 1) / 2 - sum(rank) - 1)

    print('true ranking:', gt_rank)
    print('output:', rank)
    print('correct?', np.alltrue(rank == gt_rank))

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
    print(n_comp)
    print(estnum)

    from matplotlib import pyplot as plt

    # fig, axs = plt.subplots(2)
    fig1 = plt.figure()
    plt.plot(estnum)
    fig2 = plt.figure()
    plt.plot(n_comp)

# %%
