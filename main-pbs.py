from pitsort import PITSort
from probesort import ProbeSortUC, ProbeSortULC, ProbeSortUT, ProbeSortULT
from models import WSTModel, HBTL

import numpy as np
import random
import plotly.express as px
import pandas as pd


def test_classes(n, class_list):
    delta = 0.1

    # gt_a = list(np.random.permutation(n))
    # model = WSTModel(gt_a, slackness=0.25)

    s = np.arange(1, n + 1)
    gamma = np.ones(1)
    model = HBTL(s, gamma)
    gt_a = list(np.argsort(s))

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


def run_classes(filename, run_num=0):
    random.seed(333 + run_num)
    np.random.seed(333 + run_num)
    fout = open(filename, 'w')
    classes_to_test = [PITSort, ProbeSortUC, ProbeSortULC, ProbeSortUT, ProbeSortULT]
    fout.write(",".join(map(lambda cls: cls.__name__, classes_to_test)) + "\n")
    for i in range(10, 101, 10):
        res = test_classes(i, classes_to_test)
        fout.write("{}\n".format(",".join(res)))
        fout.flush()
    fout.close()

def sched_sbatch():
    import os
    from sbatch import sbatch_template
    for i in range(100):
        fname = "output/pbs-hbtl-cmp4-{:03d}".format(i)
        args = ["python3", "main-pbs.py", "run", fname, str(i)]
        log_name = fname + ".log"
        kwargs = {
            'job_name': fname,
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
        print(f"sbatch {sbatch_name} done")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'plot':
            res = []
            col_names = None
            for i in range(100):
                fname = "output/pbs-hbtl-cmp4-{:03d}.txt".format(i)
                df = pd.read_csv(fname)
                if col_names is None:
                    col_names = df.columns.values.tolist()
                res.append(df.to_numpy())
            res = np.array(res)
            res_avg = np.average(res, axis=0).T
            res_std = np.std(res, axis=0).T
            print("avg", res_avg)
            print("std", res_std)
            x = np.arange(1, 10) * 10
            markers = ['*', 'v', 'x', 'v', 'x'] # v

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
            plt.title(f"Model")
            fig_name = f'output_plots/model.pdf'
            # plt.tight_layout()
            plt.savefig(fig_name)
            print(fig_name)
            plt.close()

        if sys.argv[1] == 's':
            sched_sbatch()

    if len(sys.argv) == 4:
        if sys.argv[1] == "run":
            filename = sys.argv[2]
            i = int(sys.argv[3])
            run_classes(filename + ".txt", i)

    if len(sys.argv) == 1:
        filename = "output/pbs-hbtl-cmp4.txt"
        # run_classes(filename, 0)
        df = pd.read_csv(filename)
        # df = pd.DataFrame()
        print(df.to_numpy())
        # fig = px.histogram(df)
        fig = px.line(df)
        # fig.show()
