# SPDX-License-Identifer: Apache-2.0
import numpy as np

class LUCBSort:
    def __init__(self, pairwise, k):
        self.k = k
        self.pairwise = pairwise  # instance of pairwise
        self.pairwise.ctr = 0  # number of random comparisons made

    def random_cmp(self, i):  # compare i to randomly choosen other item
        j = random.choice(range(self.pairwise.n - 1))
        if j >= i:
            j += 1
        return float(self.pairwise.compare(i, j))

    def alpha(self, Ti):  # confidence interval
        n = self.pairwise.n
        beta = log(n / self.delta) + 0.75 * log(log(n / self.delta)) + 1.5 * log(1 + log(Ti / 2))
        return sqrt(3 / (2 * Ti))

    def rank(self, delta=0.1, numit=6000000):
        self.delta = delta
        S = []  # list with entries ( i, T_i, scorehat_i, scorehat_i - alpha_i, scorehat_i + alpha_i, alpha_i)
        # compare each item once to initialize
        for i in range(self.pairwise.n):
            scorehat = self.random_cmp(i)
            S.append((i, 1, scorehat, scorehat - self.alpha(1), scorehat + self.alpha(1), self.alpha(1)))

        for iit in range(numit):
            # sort descending by score ('entry' has fields (i, T_i, scorehat_i,...)
            S = sorted(S, key=lambda entry: entry[2], reverse=True)
            # min scorehat_i - alpha_i; min over (1),...,(k)
            d1low = min(S[:self.k], key=lambda entry: entry[3])
            # max scorehat_i + alpha_i; max over (k+1),...,(n)
            d2up = max(S[self.k:], key=lambda entry: entry[4])

            if d1low[3] > d2up[4]:  # termination condition
                break  # terminate

            for it in [d1low, d2up]:  # items to sample in next round
                Ti = it[1] + 1
                shat = 1.0 / Ti * ((Ti - 1) * it[2] + self.random_cmp(it[0]))
                alphai = self.alpha(Ti)
                S[S.index(it)] = (it[0], Ti, shat, shat - alphai, shat + alphai, alphai)
        self.S = S
        estimated_ranking = [s[0] for s in S]
        self.ranking = [estimated_ranking[:self.k], estimated_ranking[self.k:]]

    def plot_scores(self):  # plot
        n = len(self.S)
        scorehat = [self.S[i][2] for i in range(n)]
        upperest = [self.S[i][3] for i in range(n)]
        lowerest = [self.S[i][4] for i in range(n)]
        plt.plot(range(n), scorehat, 'rx', range(n), upperest, 'bx', range(n), lowerest, 'yx')
        plt.show()

    def evaluate_perfect_recovery(self):  # did it suceed?
        return set(self.ranking[0]) == set([i for i in range(self.k)])

class BordaGapSort:
    def __init__(self, model, kset, epsilon=None):
        self.kset = kset  # k_1,..., k_{L-1}, n
        self.pairwise = pairwise  # instance of pairwise
        if epsilon == None:
            self.epsilon = 0
        else:
            self.epsilon = epsilon

    def rank(self, delta=0.1, track=0):
        '''
        track > 0 tracks every #(track) number of comparisons:
        (number of comparisons, size of active set, best estimate)
        '''
        trackdata = []
        kset = self.kset  # temporary kset
        L = len(kset)
        self.pairwise.ctr = 0

        self.S = [[] for ell in range(L)]

        # active set contains pairs (index, score estimate)
        active_set = [(i, 0.0) for i in range(self.pairwise.n)]
        kset = array(self.kset, dtype=int)
        t = 1  # algorithm time
        while len(active_set) > 0:
            # alpha = sqrt( 2*log( 1/delta) / t )
            alpha = sqrt(log(125 * self.pairwise.n * log(1.12 * t) / delta) / t)
            ## update all scores
            for ind, (i, score) in enumerate(active_set):
                j = random.choice(range(self.pairwise.n - 1))
                if j >= i:
                    j += 1
                xi = self.pairwise.compare(i, j)  # compare i to random other item
                active_set[ind] = (i, (score * (t - 1) + xi) / t)
                # track
                if track > 0:
                    if (self.pairwise.ctr % track == 0):
                        trackdata.append([self.pairwise.ctr, self.best_estimate(active_set, kset), len(active_set)])

            ## eliminate variables
            # sort descending by score
            active_set = sorted(active_set, key=lambda ind_score: ind_score[1], reverse=True)
            toremove = []

            toset = zeros(L)  # to which set did we add an index?

            # remove items
            for ind, (i, score) in enumerate(active_set):
                # determine which potential set the index falls in
                ell = 0
                while ind + 1 > kset[ell]:
                    ell += 1
                if kset[ell - 1] == 0 and kset[ell] == kset[L - 1]:  # e.g. [0 0 2] or [0 2 2] means we are done
                    self.S[ell].append(i)
                    toremove.append(ind)
                    toset[ell] += 1
                elif ell == 0 or kset[ell - 1] == 0:  # only need to check the lower bound..
                    if (score - active_set[kset[ell]][1] > alpha - self.epsilon):
                        self.S[ell].append(i)
                        toremove.append(ind)
                        toset[ell] += 1
                elif ell == L - 1 or kset[ell] == len(active_set):  # only need to check the upper bound..
                    if (active_set[kset[ell - 1] - 1][1] - score > alpha - self.epsilon):
                        self.S[ell].append(i)
                        toremove.append(ind)
                        toset[ell] += 1
                else:  # need to check both
                    if (active_set[kset[ell - 1] - 1][1] - score > alpha - self.epsilon and
                            score - active_set[kset[ell]][1] > alpha - self.epsilon):
                        self.S[ell].append(i)
                        toremove.append(ind)
                        toset[ell] += 1

            # update k:
            for ind, i in enumerate(toset):
                kset[ind:] -= int(i)

            toremove.sort()
            for ind in reversed(toremove):
                # print(t, ': del:', ind, self.epsilon)
                del active_set[ind]
            t += 1

        trackdata.append([t, len(active_set), self.best_estimate(active_set, kset), self.pairwise.ctr])
        return trackdata

    def best_estimate(self, active_set, kset):
        '''
        best estimate if we stop now..
        '''
        # sort descending by score,
        active_set = sorted(active_set, key=lambda ind_score: ind_score[1], reverse=True)
        best_S = [list(i) for i in self.S]
        best_S[0] += [i for (i, s) in active_set[0:kset[0]]]
        for ell in range(1, len(kset)):
            best_S[ell] += [i for (i, s) in active_set[kset[ell - 1]:kset[ell]]]
        return self.success_ratio(best_S)

    def evaluate_perfect_recovery(self):
        origsets = [set(range(0, self.kset[0]))]
        for i in range(1, len(self.kset)):
            origsets.append(set(range(self.kset[i - 1], self.kset[i])))
        recsets = [set(s) for s in self.S]
        return origsets == recsets

    def success_ratio(self, S=None):
        if S == None:
            S = self.S
        frac = 1.0
        for ell, ellset in enumerate(S):
            for ind in ellset:
                if ell == 0:
                    if ind <= self.kset[ell]:
                        frac -= 1.0 / self.pairwise.n
                elif ind <= self.kset[ell] and ind >= self.kset[ell - 1]:
                    frac -= 1.0 / self.pairwise.n
        return frac
