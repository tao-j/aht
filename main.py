from aht import UnevenUCBActiveRank, TwoStageSeparateRank
import random
import numpy as np
import os
import sys
import subprocess
from sbatch import sbatch_template


def run(algonum, repeat, eps=0.1, delta=0.1, N=10, M=9, gg=5.0, gb=0.5):
    random.seed(123)
    np.random.seed(123)

    # data gen method may be mentioned in Ren et al.
    thetas = []
    for i in range(100):
        li = 0.9 * np.power(1.2 / 0.8, i)
        ri = 1.1 * np.power(1.2 / 0.8, i)
        thetas.append(li + (ri - li) * (np.random.random()))
        # thetas.append(np.power(1.2 / 0.8, i))

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
            # oracle
            algo = UnevenUCBActiveRank(N, 1, delta, s, [max(gg, gb)], active=False)
        elif algonum == 2:
            # two stage
            algo = TwoStageSeparateRank(N, M, delta, s, gamma)
        elif algonum == 1:
            # act
            algo = UnevenUCBActiveRank(N, M, delta, s, gamma, active=True)
        elif algonum == 0:
            # non-act
            algo = UnevenUCBActiveRank(N, M, delta, s, gamma, active=False)
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
    n_test_range = list(range(10, 101, 10))
    m_test_range = [9, 18, 36]
    gg_range = [0.5, 1.0, 2.5]
    gb_range = [0.25, 0.5, 1.0]
    # for gb in [0.25, 1., 2.5]:
    #     for gg in [2.5, 5, 10]:
    invoker = "sequential"
    invoker = "subprocess"
    invoker = "sbatch"
    outdir = f"r{repeat}"
    outdir = "output_plots"
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    elif os.path.isfile(outdir):
        exit(1)
    outdir = "output"
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    elif os.path.isfile(outdir):
        exit(1)

    command = sys.argv[1]
    if command == "manage":
        for n in n_test_range:
            for m in m_test_range:
                for gg in gg_range:
                    for gb in gb_range:
                        for algonum in range(0, 4):
                            fname = f"n{n}-m{m}-gg{gg:.2f}-gb{gb:.2f}-algo{algonum}"
                            args = ["python3", "main.py", "run", fname]
                            log_name = os.path.join(outdir, f"{fname}.log")
                            if os.path.isfile(log_name):
                                # continue # skip when task exists
                                pass # override existing task

                            if invoker == "subprocess":
                                flog = open(log_name, 'w')
                                with flog:
                                    p = subprocess.Popen(args, stdout=flog, stderr=flog,
                                                         start_new_session=True)
                                    print(f"spawned {log_name}")

                            if invoker == "sbatch":
                                kwargs = {
                                    'job_name': fname,
                                    'file_err': log_name,
                                    'file_out': log_name,
                                    'email': '',
                                    'args': ' '.join(args)
                                }
                                sbatch = sbatch_template.format(**kwargs)
                                sbatch_name = os.path.join(outdir, f"{fname}.sbatch")
                                f = open(sbatch_name, 'w')
                                f.write(sbatch)
                                f.close()
                                os.system(f"sbatch {sbatch_name}")
                                print(f"sbatch {sbatch_name} done")

                            if invoker == "sequential":
                                print("seq started", fname)
                                avg, std = run(algonum, repeat, 0.1, delta, n, m, gg, gb)
                                print(avg, std)

    if command == "plot":
        import matplotlib.pyplot as plt
        tex_out = open("output_plots/supp_plot.tex", 'w')
        for m in m_test_range:
            tex_subfigures = []
            tex_tabulars = []
            for gb in gb_range:
                for gg in gg_range:
                    plt.figure()
                    for algonum in range(0, 4):
                        x = np.array(n_test_range)
                        y = []
                        stds = []
                        for n in n_test_range:
                            fname = f"n{n}-m{m}-gg{gg:.2f}-gb{gb:.2f}-algo{algonum}"

                            log_name = os.path.join(outdir, f"{fname}.log")
                            res_name = os.path.join(outdir, f"{fname}.txt")
                            if not os.path.isfile(log_name) or not os.path.isfile(res_name):
                                print("no", fname)
                                continue
                            logstr = open(log_name).read()
                            if len(logstr) != 0:
                                print("wrong", fname, logstr)
                            avg, std = list(map(int, open(res_name).read().split()))
                            y.append(avg)
                            stds.append(std)
                        markers = ['+', 'x', 'v', '.']
                        ax = plt.errorbar(x, y, stds, linestyle='-', marker=markers[algonum])

                    plt.legend(
                        ["Non-Adaptive User Sampling", "Adaptive User Sampling", "Two Stage Ranking", "Oracle"],
                        loc="upper left")
                    fmt = plt.ScalarFormatter()
                    ax[0].axes.yaxis.set_major_formatter(fmt)
                    plt.xlabel("Number of items to rank")
                    plt.ylabel("Sample Complexity")
                    plt.title(f"$\gamma_A = {gb}, \gamma_B = {gg}$")
                    fig_name = f'output_plots/m{m}gb{gb}gg{gg}.pdf'
                    plt.savefig(fig_name)
                    print(fig_name)
                    plt.close()

                    tex_subfigure_template = f'''\\subfigure[$\\gamma_A={gb}$, $\\gamma_B={gg}$]{{\\includegraphics[width=0.32\\textwidth]{{output_plots/m{m}gb{gb}gg{gg}.pdf}} \\label{{fig:m{m}gb{gb}gg{gg}}}}}'''
                    tex_subfigures.append(tex_subfigure_template)
            tex_subfigure_str = '\n'.join(tex_subfigures)
            tex_figs_template = f'''
            \\begin{{figure}}[H]
            \\centering
            {tex_subfigure_str}
            \\caption{{When $M = {m}$. Sample complexities v.s. number of items for all algorithms. The 3-by-3 grid shows different heterogeneous user settings where the accuracy of two group of users differs.
            \\label{{fig:exp-m{m}}}
            }}
            \\end{{figure}}
            '''
            tex_out.write(tex_figs_template)
        tex_out.close()

    elif command == "run":
        fname = sys.argv[2]
        n, m, gg, gb, algonum = sys.argv[2].split('-')
        n = int(n[1:])
        m = int(m[1:])
        gg = float(gg[2:])
        gb = float(gb[2:])
        algonum = int(algonum[4:])

        avg, std = run(algonum, repeat, 0.1, delta, n, m, gg, gb)
        f = open(os.path.join(outdir, f"{fname}.txt"), 'w')
        f.write(f"{avg} {std}")
        f.close()
