import sys
import random

from models import *


class Bandit2D:
    def __init__(self, model):
        self.model = model
        self.n = model.Pij.shape[1]
        self.pij = self.model.Pij[0]

        n = self.n
        self.W = np.zeros((n, n))
        self.C = np.zeros((n, n))

        self.regret = 0
        self.regret_div_t = 0
        self.regret_best = 0
        self.t = 0

        self.rng = np.random.default_rng(seed=42)


class Bandit2DDTSBorda(Bandit2D):
    def __init__(self, model):
        super(Bandit2DDTSBorda, self).__init__(model)
        self.b_gt = np.sum(self.pij, axis=1) / self.n
        self.i_star = np.unravel_index(np.argmax(self.b_gt, axis=None), self.b_gt.shape)
        # self.i_star, self.j_star = np.unravel_index(np.argmax(self.pij, axis=None), self.pij.shape)
        self.regret_best = np.max(self.b_gt)

        self.alpha = 1
        self.t_limit = 50000

        self.stable_count = 0
        self.last_it_jt = (None, None)

    def add_to_regret(self, i, j):
        this_regret = 2 * self.regret_best - self.b_gt[i] - self.b_gt[j]
        self.regret += this_regret
        self.regret_div_t = self.regret / self.t
        return this_regret

    def loop(self):
        n = self.n
        W = self.W
        C = self.C

        self.stable_count = 0
        for t in range(1, self.t_limit):
            self.t = t
            zero_idx = C == 0
            mu = W / (C + 1e-6)
            mu[zero_idx] = 0.5
            cb = np.sqrt(self.alpha * np.log(t) / (C + 1e-6))
            cb[zero_idx] = 0.5

            mu_lcb = mu - cb
            mu_ucb = mu + cb
            # zeta_lcb = np.sum(mu_lcb, axis=1)/n
            zeta_ucb = np.sum(mu_ucb, axis=1) / n
            # zeta_lcb_max = np.max(zeta_lcb)
            zeta_max = np.max(np.sum(mu, axis=1) / n)
            zeta_remove_i = zeta_ucb < zeta_max
            # zeta_sel_i = zeta_ucb > zeta_max

            theta = self.rng.beta(W + 1, C - W + 1)
            np.fill_diagonal(theta, 0.5)
            b = np.sum(theta, axis=1) / n
            b[zeta_remove_i] = 0
            if np.any(zeta_remove_i):
                print('removed', zeta_remove_i)
            i_t = np.argmax(b)

            # # i_t winning prob.
            # phi = self.rng.beta(W[i_t, :] + 1, C[i_t, :] - W[i_t, :] + 1)
            # # ignore items must worse than i_t
            # phi[mu_lcb[i_t] > 0.5] = 1
            # phi[i_t] = 0.5
            # # item that lose to i_t with the lowest prob.
            # # (potentially <0.5 then it actually beats i_t, otherwise pick i_t instead)
            # j_t = np.argmin(phi)

            # i_t losing prob.
            phi = self.rng.beta(W[:, i_t] + 1, C[:, i_t] - W[:, i_t] + 1)
            # ignore items must better than i_t
            phi[mu_lcb[:, i_t] > 0.5] = 0
            phi[i_t] = 0.5
            # item that could beat i_t with the highest prob.
            # (>0.5 then it beats i_t, otherwise pick i_t instead)
            j_t = np.argmax(phi)


            y_t = self.model.sample_pair(i_t, j_t)
            C[i_t, j_t] += 1
            C[j_t, i_t] += 1
            W[i_t, j_t] += y_t
            W[j_t, i_t] += 1 - y_t
            this_regret = self.add_to_regret(i_t, j_t)

            self.last_it_jt = (i_t, j_t)
            if self.last_it_jt == (self.i_star, self.i_star):
                self.stable_count += 1
            else:
                self.stable_count = 0
            if self.stable_count > 100:
                break
            if t % 100 == 0:
                print("{:.3f} ".format(self.regret_div_t), end="")
                print(t, i_t, j_t, this_regret, "                    ", end='\r')
                sys.stdout.flush()
        print()
        sii = np.sum([C[i, i] if i != self.i_star[0] else 0 for i in range(n)])
        sii_star = C[self.i_star, self.i_star]
        print("ii", sii)
        print("i*i*", sii_star)
        print("ij", np.sum(C) - sii - sii_star)
        return self.last_it_jt, self.i_star, self.t


if __name__ == "__main__":
    n = 15
    np.random.seed(42)
    random.seed(42)

    for mdl_cls in [WSTModel, HBTL, WSTAdjModel, AdjacentSqrtModel]:
        model = mdl_cls(np.random.permutation(np.arange(0, n)))
        # print(model.Pij)

        tsb = Bandit2DDTSBorda(model)
        print(mdl_cls.__name__, tsb.loop())
        print("---------------------------")
