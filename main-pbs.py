from pitsort import PITSort
from probesort import ProbeSortUC, ProbeSortULC, ProbeSortUT, ProbeSortULT
from models import WSTModel, HBTL, FixedModel, AdjacentOnlyModel

import os
import numpy as np
import random
import pandas as pd


def test_classes(n, class_list, model, delta_d):
    delta = 0.10

    if model == "wst":
        gt_a = list(np.random.permutation(n))
        model = WSTModel(gt_a, slackness=delta_d)
    elif model == "adj":
        gt_a = list(np.random.permutation(n))
        model = AdjacentOnlyModel(gt_a, slackness=delta_d)
    elif model == "hbtl":
        s = np.arange(1, n + 1) * np.sqrt(delta_d) * 2
        # TODO: Uniform([0.9 ∗ 1.5^n−i , 1.1 ∗ 1.5^n−i ])
        s = np.random.permutation(s)
        gamma = np.ones(1)
        model = HBTL(s, gamma)
        gt_a = list(np.argsort(s))
    elif model == "fixed":
        gt_a = list(np.random.permutation(n))
        model = FixedModel(gt_a, slackness=delta_d)
    else:
        raise NotImplementedError

    res = []
    print("n:", n, gt_a)
    for cls in class_list:
        cls_s = cls(n, delta, model)

        if cls == PITSort and model_str == "adj":
            cls_n = 1400000
        else:
            cls_a = cls_s.arg_sort()
            cls_n = cls_s.sample_complexity
            assert gt_a == cls_a

        print(cls.__name__, cls_n)
        # print("rnk", cls_a)
        res.append(str(cls_n))

    return res


def run_classes(filename, model_str, run_num=0, max_n=100, delta_d="0.25"):
    delta_d = float(delta_d)
    random.seed(333 + run_num)
    np.random.seed(333 + run_num)
    fout = open(filename, 'w')
    classes_to_test = [PITSort, ProbeSortUC, ProbeSortULC, ProbeSortUT, ProbeSortULT]
    fout.write(",".join(map(lambda cls: cls.__name__, classes_to_test)) + "\n")
    for i in range(10, max_n + 1, 10):
        res = test_classes(i, classes_to_test, model_str, delta_d)
        fout.write("{}\n".format(",".join(res)))
        fout.flush()
    fout.close()


def get_fname(model_str, i, delta_d):
    return "output-{}/pbs-{}-cmp4-{:03d}".format(delta_d, model_str, i)


def submit_sbatch(model_str, max_n, repeat, delta_d):
    from sbatch import sbatch_template
    for i in range(repeat):
        fname = get_fname(model_str, i, delta_d)
        args = ["python3", "main-pbs.py", "run",
                fname, str(i), str(max_n), model_str, delta_d]
        log_name = fname + ".log"
        kwargs = {
            'job_name': "pbs{:03d}{}{}".format(i, model_str, delta_d),
            'file_err': log_name,
            'file_out': log_name,
            'email': '',
            'args': ' '.join(args)
        }
        sbatch = sbatch_template.format(**kwargs)
        sbatch_name = os.path.join(f"{fname}.sbatch")
        f = open(sbatch_name, 'w')
        f.write(sbatch)
        f.close()
        os.system(f"sbatch {sbatch_name}")
        os.system(f"rm {sbatch_name}")
        print(f"sbatch {sbatch_name} done")


def plot_mat(model_str, max_n, repeat, delta_d):
    res = []
    col_names = None
    for i in range(repeat):
        fname = get_fname(model_str, i, delta_d=delta_d)
        df = pd.read_csv(fname + ".txt")
        if col_names is None:
            col_names = df.columns.values.tolist()
        res.append(df.to_numpy())
        # TODO: adj sometime fails
        if res[-1].shape != res[0].shape:
            print(i, res[-1].shape)
            res.pop()
            continue
    res = np.array(res)

    res_avg = np.average(res, axis=0).T
    res_std = np.std(res, axis=0).T
    # print("avg", res_avg)
    # print("std", res_std)
    x = np.arange(10, max_n + 1, 10)
    markers = ['*', 'v', 'x', 'v', 'x']

    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 14})

    pitsort_idx = -1
    for i in range(len(col_names)):
        if res_avg[i].shape != x.shape:
            print(res_avg.shape, x.shape)
            continue
        if col_names[i] == PITSort.__name__ and model_str == "adj":
            pitsort_idx = i
            ax = plt.errorbar(x, res_avg[i], res_std[i],
                              uplims=False, lolims=False, linestyle='None',
                              marker='None')
        else:
            ax = plt.errorbar(x, res_avg[i], res_std[i],
                              uplims=False, lolims=False, linestyle='-',
                              marker=markers[i % 5])
    if model_str == "adj":
        plt.legend(col_names[:pitsort_idx] +
                   [PITSort.__name__ + " (failed)"] +
                   col_names[pitsort_idx + 1:], loc="lower right")
    else:
        plt.legend(col_names, loc="lower right")
    fmt = plt.ScalarFormatter()
    ax[0].axes.yaxis.set_major_formatter(fmt)
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel("Number of items to rank")
    plt.ylabel("Sample complexity")
    # plt.ylim(bottom=0, top=800000)
    model_str_to_setting = {
        'hbtl': "SST",
        'wst': "WST",
        'adj': "AdjacentOnly",
        'fixed': "Uniform",
    }
    setting = model_str_to_setting[model_str]
    plt.title(f"Performance in \\verb+{setting}+ setting. $\Delta_d = {delta_d}$")
    fig_name = f'output_plots/{model_str}-n{max_n}x{repeat}-{delta_d}.pdf'
    plt.tight_layout()
    plt.savefig(fig_name)

    print(fig_name)
    plt.close()
    os.system(f"pdfcrop {fig_name} {fig_name}")
    os.system(f"pdffonts {fig_name} | grep 'Type 3'")


if __name__ == "__main__":
    repeat = 100
    max_n = 100
    model_strs = ["hbtl", "wst", "fixed", "adj"]
    # model_strs = ["fixed"]

    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'plot':
            for delta_d in ["0.25", "0.10"]:
                for model_str in model_strs:
                    plot_mat(model_str, max_n, repeat, delta_d)

        if sys.argv[1] == 's':
            for model_str in model_strs:
                for delta_d in ["0.25", "0.10"]:
                    submit_sbatch(model_str, max_n, repeat, delta_d)
            os.system(f"watch 'squeue -o \"%.18i %.9P %.18j %.8u %.2t %.10M %.6D %R\" | grep pbs'")

    if len(sys.argv) == 7:
        if sys.argv[1] == "run":
            filename = sys.argv[2]
            i = int(sys.argv[3])
            max_n = int(sys.argv[4])
            model_str = sys.argv[5]
            delta_d = sys.argv[6]
            run_classes(filename + ".txt",
                        model_str, run_num=i, max_n=max_n, delta_d=delta_d)

    if len(sys.argv) == 1:
        model_str = 'wst'
        filename = get_fname(model_str, 0, delta_d=0.25) + "-s.txt"
        run_classes(filename, model_str, run_num=10, max_n=10)

        import plotly.express as px

        df = pd.read_csv(filename)
        print(df.to_numpy())
        fig = px.line(df)
        # fig.show()
