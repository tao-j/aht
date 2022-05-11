from pitsort import PITSort
from probesort import ProbeSortUC, ProbeSortULC, ProbeSortUT, ProbeSortULT
from models import WSTModel, HBTL, Uniform, AdjacentOnlyModel

import numpy as np
import random
import pandas as pd


def test_classes(n, class_list, model):
    delta = 0.1

    if model == "wst":
        gt_a = list(np.random.permutation(n))
        model = WSTModel(gt_a, slackness=0.25)
    elif model == "adj":
        gt_a = list(np.random.permutation(n))
        model = AdjacentOnlyModel(gt_a, slackness=0.25)
    elif model == "hbtl":
        s = np.arange(1, n + 1)
        gamma = np.ones(1)
        model = HBTL(s, gamma)
        gt_a = list(np.argsort(s))
    elif model == "uni":
        s = np.arange(1, n + 1)
        gamma = np.ones(1)
        model = Uniform(s, gamma)
        gt_a = list(np.argsort(s))
    else:
        raise NotImplementedError

    res = []
    print("n:", n, gt_a)
    for cls in class_list:
        cls_s = cls(n, delta, model)

        cls_a = cls_s.arg_sort()
        cls_n = cls_s.sample_complexity
        print(cls.__name__, cls_n)
        # print("rnk", cls_a)

        assert gt_a == cls_a
        res.append(str(cls_n))

    return res


def run_classes(filename, model_str, run_num=0, max_n=100):
    random.seed(333 + run_num)
    np.random.seed(333 + run_num)
    fout = open(filename, 'w')
    classes_to_test = [PITSort, ProbeSortUC, ProbeSortULC, ProbeSortUT, ProbeSortULT]
    fout.write(",".join(map(lambda cls: cls.__name__, classes_to_test)) + "\n")
    for i in range(10, max_n + 1, 10):
        res = test_classes(i, classes_to_test, model_str)
        fout.write("{}\n".format(",".join(res)))
        fout.flush()
    fout.close()


def get_fname(model_str, i):
    return "output/pbs-{}-cmp4-{:03d}".format(model_str, i)


def submit_sbatch(model_str, max_n, repeat):
    import os
    from sbatch import sbatch_template
    for i in range(repeat):
        fname = get_fname(model_str, i)
        args = ["python3", "main-pbs.py", "run", fname, str(i), str(max_n), model_str]
        log_name = fname + ".log"
        kwargs = {
            'job_name': "pbs{:03d}{}".format(i, model_str),
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


def plot_mat(model_str, max_n, repeat):
    res = []
    col_names = None
    for i in range(repeat):
        fname = get_fname(model_str, i)
        df = pd.read_csv(fname + ".txt")
        if col_names is None:
            col_names = df.columns.values.tolist()
        res.append(df.to_numpy())
    res = np.array(res)

    res_avg = np.average(res, axis=0).T
    res_std = np.std(res, axis=0).T
    # print("avg", res_avg)
    # print("std", res_std)
    x = np.arange(10, max_n + 1, 10)
    markers = ['*', 'v', 'x', 'v', 'x']

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams['text.usetex'] = True
    for i in range(len(col_names)):
        ax = plt.errorbar(x, res_avg[i], res_std[i], uplims=False, lolims=False, linestyle='-',
                          marker=markers[i % 5])
    plt.legend(col_names, loc="upper left")
    fmt = plt.ScalarFormatter()
    ax[0].axes.yaxis.set_major_formatter(fmt)
    plt.xlabel("Number of items to rank")
    plt.ylabel("Sample complexity")
    plt.ylim(bottom=0, top=800000)
    plt.title(f"Performance comparison in {model_str.upper()} setting")
    fig_name = f'output_plots/{model_str}-n{max_n}x{repeat}.pdf'
    # plt.tight_layout()
    plt.savefig(fig_name)
    print(fig_name)
    plt.close()


if __name__ == "__main__":
    repeat = 100
    max_n = 100
    model_strs = ["wst", "hbtl", "uni"]

    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'plot':
            for model_str in model_strs:
                plot_mat(model_str, max_n, repeat)

        if sys.argv[1] == 's':
            for model_str in model_strs:
                submit_sbatch(model_str, max_n, repeat)
            import os

            os.system(f"watch 'squeue | grep pbs'")

    if len(sys.argv) == 6:
        if sys.argv[1] == "run":
            filename = sys.argv[2]
            i = int(sys.argv[3])
            max_n = int(sys.argv[4])
            model_str = sys.argv[5]
            run_classes(filename + ".txt", model_str, run_num=i, max_n=max_n)

    if len(sys.argv) == 1:
        model_str = 'adj'
        filename = get_fname(model_str, 0) + "-s.txt"
        run_classes(filename, model_str, run_num=10, max_n=100)

        import plotly.express as px
        df = pd.read_csv(filename)
        print(df.to_numpy())
        fig = px.histogram(df)
        fig = px.line(df)
        fig.show()
