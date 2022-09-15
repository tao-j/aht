# SPDX-License-Identifer: GPL-3.0-only
import sys
import random
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

from models import *


class Bandit2D:
    def __init__(self, model, seed, t_limit=1000):
        self.model = model
        self.n = model.Pij.shape[1]
        # TODO: take care of 3D matrix
        self.pij = self.model.Pij[0]

        n = self.n
        # number of times ij wins
        self.W = np.zeros((n, n))
        # total number of times this pair is pulled
        # C_ij == C_ji
        self.C = np.zeros((n, n))

        self.regret = 0
        self.regret_div_t = 0
        self.regret_best = 0
        self.t = 0

        self.rng = np.random.default_rng(seed=seed)

        self.r_gt = self.r_metric(self.pij)
        self.i_star = np.unravel_index(np.argmax(self.r_gt, axis=None), self.r_gt.shape)
        # self.i_star, self.j_star = np.unravel_index(np.argmax(self.pij, axis=None), self.pij.shape)
        self.regret_best = np.max(self.r_gt)

        self.alpha = 1
        self.t_limit = t_limit

        self.stable_count = 0
        self.last_it_jt = (None, None)

    def r_metric(self, a):
        raise NotImplementedError

    def r_filter(self, mu, mu_ucb, mu_lcb):
        raise NotImplementedError

    def pick_pair(self, t):
        raise NotImplementedError

    def loop(self):
        self.stable_count = 0
        self.regret_list = []
        for t in range(1, self.t_limit):
            i_t, j_t = self.pick_pair(t)

            y_t = self.model.sample_pair(i_t, j_t)
            self.W[i_t, j_t] += y_t
            self.W[j_t, i_t] += 1 - y_t
            self.C = self.W + self.W.T
            this_regret = self.add_to_regret(i_t, j_t)

            self.last_it_jt = (i_t, j_t)
            if self.last_it_jt == (self.i_star, self.i_star):
                self.stable_count += 1
            else:
                self.stable_count = 0
            # if self.stable_count > 1000:
            #     break
            if t % 10 == 0:
                print("{:.3f} ".format(self.regret_div_t), end="")
                print(t, i_t, j_t, this_regret, "                    ", end="\r")
                sys.stdout.flush()
                self.regret_list.append(self.regret)
        print()

        plt.plot(
            np.arange(0, len(self.regret_list)) * 10,
            self.regret_list,
            label=self.model.__class__.__name__,
        )

        sii = np.sum([self.W[i, i] if i != self.i_star[0] else 0 for i in range(n)])
        sii_star = self.W[self.i_star, self.i_star]
        print("ii", sii)
        print("i*i*", sii_star)
        print("ij", np.sum(self.W) - sii - sii_star)

        return self.last_it_jt, self.i_star, self.t

    def save_plot(self, seed):
        plt.title(self.__class__.__name__)
        # plt.show()
        plt.legend(loc="upper left")
        plt.savefig("output_fig/" + self.__class__.__name__ + f"-{seed}.png")

    def add_to_regret(self, i, j):
        this_regret = 2 * self.regret_best - self.r_gt[i] - self.r_gt[j]
        self.regret += this_regret
        self.regret_div_t = self.regret / self.t
        return this_regret


class ITS(Bandit2D):
    def pick_pair(self, t):
        self.t = t
        zero_idx = self.C == 0
        mu = self.W / (self.C + 1e-6)
        mu[zero_idx] = 0.5

        ret = [None, None]
        for i in range(2):
            theta = self.rng.beta(self.W + 1, self.C - self.W + 1)
            np.fill_diagonal(theta, 0.5)
            r_hat = self.r_metric(theta)
            ret[i] = np.argmax(r_hat)

        return ret[0], ret[1]


class DTS(Bandit2D):
    def pick_pair(self, t):
        self.t = t
        zero_idx = self.C == 0
        mu = self.W / (self.C + 1e-6)
        mu[zero_idx] = 0.5
        cb = np.sqrt(self.alpha * np.log(t) / (self.C + 1e-6))
        cb[zero_idx] = 0.5

        mu_lcb = mu - cb
        mu_ucb = mu + cb

        zeta_remove_i = self.r_filter(mu, mu_ucb, mu_lcb)

        theta = self.rng.beta(self.W + 1, self.C - self.W + 1)
        np.fill_diagonal(theta, 0.5)
        r_hat = self.r_metric(theta)
        r_hat[zeta_remove_i] = 0
        i_t = np.argmax(r_hat)

        # i_t losing prob.
        phi = self.rng.beta(self.W[:, i_t] + 1, self.C[:, i_t] - self.W[:, i_t] + 1)
        # ignore items must better than i_t
        phi[mu_lcb[:, i_t] > 0.5] = 0
        phi[i_t] = 0.5
        j_t = np.argmax(phi)

        return i_t, j_t


class DUCB(Bandit2D):
    def pick_pair(self, t):
        self.t = t
        zero_idx = self.C == 0
        mu = self.W / (self.C + 1e-6)
        mu[zero_idx] = 0.5
        cb = np.sqrt(self.alpha * np.log(t) / (self.C + 1e-6))
        cb[zero_idx] = 0.5

        zeta = np.sum(mu, axis=1) / self.n
        tmp_cb = self.alpha * np.log(t) / (self.C + 1e-6)
        tmp_cb[zero_idx] = 0.5
        zeta_cb = np.sqrt(np.sum(tmp_cb, axis=1) / self.n)

        i_t = np.argmax(zeta + zeta_cb)
        j_t = np.argmax(zeta + 1 * cb[i_t, :])

        return i_t, j_t


class TraitEmpirical:
    def pick_jt(self, r_hat, cb, i_t, zeta_cb, zeta):
        # Single
        j_t = np.argmax(zeta + 1 * cb[i_t, :])
        return j_t


class TraitSingle:
    def pick_jt(self, r_hat, cb, i_t, zeta_cb, zeta):
        # Single
        j_t = np.argmax(r_hat + 1 * cb[i_t, :])
        return j_t


class TraitAll:
    def pick_jt(self, r_hat, cb, i_t, zeta_cb, zeta):
        # All
        test = r_hat[i_t] - zeta_cb[i_t] > r_hat + zeta_cb
        test[i_t] = True
        if np.all(test):
            j_t = i_t
        else:
            j_t = self.rng.choice(self.n)
        return j_t


class DBD(Bandit2D):
    def pick_jt(self, *argv):
        raise NotImplementedError

    def pick_pair(self, t):
        self.t = t
        zero_idx = self.C == 0
        mu = self.W / (self.C + 1e-6)
        mu[zero_idx] = 0.5
        # cb = np.sqrt(self.alpha * np.log(t) / (self.C + 1e-6))
        # cb[zero_idx] = 0.5
        cb = np.sqrt(self.alpha * np.log(t) / (self.C + 1e-6))
        cb[zero_idx] = 5

        mu_lcb = mu - cb
        mu_ucb = mu + cb

        zeta = np.sum(mu, axis=1) / self.n
        tmp_cb = self.alpha * np.log(t) / (self.C + 1e-6)
        tmp_cb[zero_idx] = 0.5
        zeta_cb = np.sqrt(np.sum(tmp_cb, axis=1) / self.n)

        zeta_remove_i = self.r_filter(mu, mu_ucb, mu_lcb)

        theta = self.rng.beta(self.W + 1, self.C - self.W + 1)
        np.fill_diagonal(theta, 0.5)
        r_hat = self.r_metric(theta)
        r_hat[zeta_remove_i] = 0
        # if np.any(zeta_remove_i):
        #     print('removed', zeta_remove_i)
        i_t = np.argmax(r_hat)

        j_t = self.pick_jt(r_hat, cb, i_t, zeta_cb, zeta)
        return i_t, j_t


class DBDAll(TraitAll, DBD):
    pass


class DBDSingle(TraitSingle, DBD):
    pass


class DBDSingleEmpirical(TraitEmpirical, DBDSingle):
    pass


class Copland(Bandit2D):
    def r_metric(self, a):
        return np.sum(a > 0.5, axis=1) / self.n

    def r_filter(self, mu, mu_ucb, mu_lcb):
        hat_zeta = np.sum(mu_ucb > 0.5, axis=1) / self.n
        hat_zeta_max = np.max(hat_zeta)
        return hat_zeta != hat_zeta_max


class Borda(Bandit2D):
    def r_metric(self, a):
        return np.sum(a, axis=1) / self.n

    def r_filter(self, mu, mu_ucb, mu_lcb):
        return np.ones(self.n) < 0


class ITSCopland(Copland, ITS):
    pass


class ITSBorda(Borda, ITS):
    pass


class DTSCopland(Copland, DTS):
    pass


class DTSBorda(Borda, DTS):
    pass


class DUCBBorda(Borda, DUCB):
    pass


class DBDBordaAll(Borda, DBDAll):
    pass


class DBDBordaSingle(Borda, DBDSingle):
    pass


class DBDBordaSingleEmpirical(Borda, DBDSingleEmpirical):
    pass


if __name__ == "__main__":
    n = 15
    import time

    seed = int(time.time())
    seed = 43
    np.random.seed(seed)
    random.seed(seed)

    for mdl_cls in [
        WSTAdjModel,
        AdjacentSqrtModel,
        CountryPopulationNoUser,
        WSTModel,
        Rand,
    ]:
        model = mdl_cls(np.random.permutation(np.arange(0, n)))
        # model = mdl_cls((np.arange(0, n)))
        # print(model.Pij)

        # tsb = ITSCopland(model, seed)
        # tsb = ITSBorda(model, seed)
        tsb = DTSCopland(model, seed)
        # tsb = DTSBorda(model, seed)
        # tsb = DBDBordaAll(model, seed)
        # tsb = DBDBordaSingle(model, seed)
        # tsb = DBDBordaSingleEmpirical(model, seed)
        # tsb = DUCBBorda(model, seed)

        tsb.t_limit = 200000
        print(mdl_cls.__name__, tsb.loop())
        print("---------------------------")

    tsb.save_plot(seed)
