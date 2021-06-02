from aht import UnevenUCBActiveRank, TwoStageSimultaneousActiveRank, TwoStageSeparateRank
import random
import numpy as np

def gamma_sweep(algonum, repeat, eps=0.1, delta=0.1, N=10, gg=5.0):
    random.seed(123)
    np.random.seed(123)

    # data gen method may be mentioned in Ren et al.
    thetas = []
    for i in range(100):
        li = 0.9 * np.power(1.2 / 0.8, i)
        ri = 1.1 * np.power(1.2 / 0.8, i)
        thetas.append(li + (ri - li) * (np.random.random()))
        # thetas.append(np.power(1.2 / 0.8, i))

    M = 9
    # for gb in [0.25, 1., 2.5]:
    #     for gg in [2.5, 5, 10]:
    # for gb in [2.5]:
    #     for gg in [2.5]:
    for gb in [0.5]:
        for gg in [gg]:
            gamma = [gg] * (M // 3) + [gb] * (M // 3 * 2)
            gamma += [gb] * (M - len(gamma))
            # s = np.linspace(1 / n, 1, n)
            s = np.log(thetas[:N])
            tts = []

            for _ in range(repeat):
                s_idx = list(range(0, len(s)))
                random.shuffle(s_idx)
                s = s[s_idx]
                # np.random.shuffle(s)

                # algo = TwoStageSimultaneousActiveRank(N, M, delta, s, gamma)
                if algonum == 3:
                    algo = UnevenUCBActiveRank(N, 1, delta, s, [gg], active=False)
                elif algonum == 2:
                    algo = TwoStageSeparateRank(N, M, delta, s, gamma)
                else:
                    algo = UnevenUCBActiveRank(N, M, delta, s, gamma, active=algonum)
                rank_sample_complexity, arg_list = algo.rank()
                tts.append(rank_sample_complexity)
                a_ms = list(s[arg_list])
                a_sorted = sorted(s)

                assert (a_ms == a_sorted)
                # print("selected users", algo.cU)
            return int(np.average(tts)), int(np.std(tts))


if __name__ == "__main__":
    repeat = 100
    delta = 0.2
    # for delta in np.arange(0.05, 1, 0.05):
    test_range = range(10, 11, 10)
    i = 1
    for gg in [1.0]:
        print(f"{gg} ----")
        ox = []
        oy = []
        oz = []
        oo = []
        for n in test_range:
            print("Items", n)
            ox.append(gamma_sweep(algonum=0, repeat=repeat, delta=delta, N=n, gg=gg))
            oy.append(gamma_sweep(algonum=1, repeat=repeat, delta=delta, N=n, gg=gg))
            oz.append(gamma_sweep(algonum=2, repeat=repeat, delta=delta, N=n, gg=gg))
            oo.append(gamma_sweep(algonum=3, repeat=repeat, delta=delta, N=n, gg=gg))

    # return
        import os

        os.chdir("{:.1f}".format(gg))
        fmt = lambda x: '{},{}'.format(x[0], x[1])
        def save_output(fname, param):
            fout = open(f"{fname}.txt", 'w')
            fout.write('\n'.join(list(map(fmt, param))))
            fout.close()
        save_output("aht-non-act", ox)
        save_output("aht-act", oy)
        save_output("aht-staged", oz)
        save_output("aht-oracle", oo)

        import matplotlib.pyplot
        import matplotlib.pyplot as plt
        import numpy as np

        def plot_output(fname, test_range):
            x = np.array(test_range)
            y = []
            std = []
            fin = open(fname + ".txt")
            for line in fin.readlines():
                ax, astd = line.split(',')
                y.append(int(ax))
                std.append(int(astd))
            ax = plt.errorbar(x, y, std, linestyle='-', marker='x')
            fin.close()
            return ax

        plt.figure()
        plot_output("aht-non-act", test_range)
        plot_output("aht-act", test_range)
        plot_output("aht-staged", test_range)
        ax = plot_output("aht-oracle", test_range)

        plt.legend(["Non-Adaptive User Sampling", "Adaptive User Sampling", "Two Stage Ranking", "Oracle"])
        fmt = matplotlib.pyplot.ScalarFormatter()
        ax[0].axes.yaxis.set_major_formatter(fmt)
        plt.xlabel("Number of items to rank")
        plt.ylabel("Sample Complexity")

        plt.title("$\gamma_A = 0.5, \gamma_B = {:.1f}$".format(gg))
        plt.savefig(f'nonacac{i}.pdf')
        os.chdir("../")
        i = i + 1
