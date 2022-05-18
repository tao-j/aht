from pitsort import PITSort
from probesort import ProbeSortUC, ProbeSortULC, ProbeSortUT, ProbeSortULT
from models import WSTModel, HBTL, WSTAdjModel, AdjacentOnlyModel, AdjacentSqrtModel, AdjacentConstantModel

import os
import numpy as np
import random
import pandas as pd

model_str_to_setting = {
    'sst': "\\verb+   SST+",
    'wst': "\\verb+   WST+",
    'adj': "\\verb+  ADJ+$\gamma_1$",
    'ads': "\\verb+  ADJ+$\gamma_2$",
    'adc': "\\verb+ADJ CNST+",
    'wstadj': "\\verb+WSTADJ+",
}

col_names_mapping = {
    PITSort.__name__: "IIR",
    ProbeSortUT.__name__: "Probe-Rank",
    ProbeSortUC.__name__: "Probe-Rank-Opt"
}


def test_classes(n, class_list, model, delta_d):
    delta = 0.10

    if model == "wst":
        gt_a = list(np.random.permutation(n))
        model = WSTModel(gt_a, delta_d=delta_d)
    elif model == "adj":
        gt_a = list(np.random.permutation(n))
        model = AdjacentOnlyModel(gt_a, delta_d=delta_d)
    elif model == "ads":
        gt_a = list(np.random.permutation(n))
        model = AdjacentSqrtModel(gt_a, delta_d=delta_d)
    elif model == "adc":
        gt_a = list(np.random.permutation(n))
        model = AdjacentConstantModel(gt_a, delta_d=delta_d)
    elif model == "sst":
        s = np.arange(1, n + 1) * delta_d * 100 / n
        # TODO: Uniform([0.9 ∗ 1.5^n−i , 1.1 ∗ 1.5^n−i ])
        s = np.random.permutation(s)
        gamma = np.ones(1)
        model = HBTL(s, gamma)
        gt_a = list(np.argsort(s))
    elif model == "wstadj":
        gt_a = list(np.random.permutation(n))
        model = WSTAdjModel(gt_a, delta_d=delta_d)
    else:
        raise NotImplementedError

    res = []
    print("n:", n, gt_a)
    for cls in class_list:
        cls_s = cls(n, delta, model)

        cls_a = cls_s.arg_sort()
        cls_n = cls_s.sample_complexity
        if cls_a is not None:
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
    classes_to_test = [PITSort, ProbeSortUC, ProbeSortUT]
    fout.write(",".join(map(lambda cls: cls.__name__, classes_to_test)) + "\n")
    for i in range(10, max_n + 1, 10):
        res = test_classes(i, classes_to_test, model_str, delta_d)
        fout.write("{}\n".format(",".join(res)))
        fout.flush()
    fout.close()


def get_fname(model_str, i, delta_d):
    return "output/pbs-{}-cmp4-{:03d}-{}".format(model_str, i, delta_d)


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


def plot_deltad(model_str, max_n, repeat, delta_d):
    res = []
    col_names = None
    for i in range(repeat):
        fname = get_fname(model_str, i, delta_d=delta_d)
        try:
            df = pd.read_csv(fname + ".txt")
        except:
            print(fname, "bad ========================")
        if col_names is None:
            col_names = df.columns.values.tolist()
        res.append(df.to_numpy())
        if res[-1].shape != res[0].shape:
            print(i, res[-1].shape, "rank failure")
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
    plt.rcParams.update({'font.size': 27})

    for i in range(len(col_names)):
        if res_avg[i].shape != x.shape:
            print(res_avg.shape, x.shape)
            continue
        if i == 1:
            continue
        ax = plt.errorbar(x, res_avg[i], res_std[i],
                          uplims=False, lolims=False, linestyle='-',
                          marker=markers[i % 5])
    cc = []
    for i in range(len(col_names)):
        if i == 1:
            continue
        cc.append(col_names_mapping[col_names[i]])
    # plt.legend(col_names, loc="lower right")
    plt.legend(cc, prop={'size': 22})
    fmt = plt.ScalarFormatter()
    ax[0].axes.yaxis.set_major_formatter(fmt)
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel("Number of items to rank")
    plt.ylabel("Sample complexity")
    # plt.ylim(bottom=10**0, top=10**10)
    setting = model_str_to_setting[model_str]
    # plt.title(f"$\Delta_d = {delta_d}$ {setting} model")
    fig_name = f'output_plots/{model_str}-n{max_n}x{repeat}-{delta_d}.pdf'
    plt.tight_layout()
    plt.savefig(fig_name)

    print(fig_name)
    plt.close()
    os.system(f"pdfcrop {fig_name} {fig_name} > /dev/null")
    os.system(f"pdffonts {fig_name} | grep 'Type 3'")


def plot_n(model_str, n, repeat, delta_ds):
    res = []
    col_names = None
    for d_i, delta_d in enumerate(delta_ds):
        des = []
        for i in range(repeat):
            fname = get_fname(model_str, i, delta_d=delta_d)
            try:
                df = pd.read_csv(fname + ".txt")
            except:
                pass
                # print(fname, "bad ========================")
            if col_names is None:
                col_names = df.columns.values.tolist()
            else:
                assert col_names == df.columns.values.tolist()
            # TODO: hacky here
            try:
                des.append(df.to_numpy()[n // 10 - 1].tolist())
                if np.any(np.array(des[-1]) > 10**9):
                    print(des[-1], delta_d)
            except:
                print("---- ---- failed one run", model_str, delta_d, i)
                des.append(des[-1])
        res.append(des)
    res_avg = np.average(res, axis=1)
    res_std = np.std(res, axis=1)
    # print("avg", res_avg)
    # print("std", res_std)
    x = np.array(list(map(float, delta_ds)))

    markers = ['*', 'v', 'x', 'v', 'x']

    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 27})

    for i in range(len(col_names)):
        if res_avg[:, i].shape != x.shape:
            print(i, res_avg.shape, x.shape)
            continue
        if i == 1:
            continue
        ax = plt.errorbar(x, res_avg[:, i], res_std[:, i],
                          uplims=False, lolims=False, linestyle='-',
                          marker=markers[i % 5])
    cc = []
    for i in range(len(col_names)):
        if i == 1:
            continue
        cc.append(col_names_mapping[col_names[i]])
    # plt.legend(col_names, loc="lower right")
    plt.legend(cc, prop={'size': 22})
    fmt = plt.ScalarFormatter()
    ax[0].axes.yaxis.set_major_formatter(fmt)
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel("$\Delta_d$")
    plt.ylabel("Sample complexity")
    # plt.ylim(bottom=10**0, top=10**10)

    setting = model_str_to_setting[model_str]
    # plt.title(f"$n = {n}$ {setting} model")
    fig_name = f'output_plots/dd-{model_str}-n{n:03d}x{repeat}.pdf'
    plt.tight_layout()
    plt.savefig(fig_name)

    print(fig_name)
    plt.close()
    os.system(f"pdfcrop {fig_name} {fig_name} > /dev/null")
    os.system(f"pdffonts {fig_name} | grep 'Type 3'")


if __name__ == "__main__":
    repeat = 100
    max_n = 100
    # model_strs = ["sst", "wst", "wstadj", "adj", "ads", "adc"]
    # model_strs = ["sst", "wst", "adj", "ads"]
    # model_strs = ["wstadj", "adc"]
    # model_strs = ["ads"]
    # model_strs = ["adc"]
    model_strs = ["sst"]

    import sys

    delta_ds = ["0.40", "0.30", "0.20", "0.10"]
    # delta_ds = ["0.40", "0.35", "0.30", "0.25", "0.20", "0.15", "0.10", "0.05"]
    # delta_ds = ["0.40", "0.35", "0.30", "0.25", "0.20", "0.15", "0.10", "0.05", "0.01"]
    # delta_ds = ["0.01"]
    if len(sys.argv) > 1:
        if sys.argv[1] == 'plot':
            os.system("grep -i error output/*")
            os.system("grep -i assert output/*")
            for n in [20, 40, 60, 80]:
                for model_str in model_strs:
                    try:
                        plot_n(model_str, n, repeat, delta_ds)
                    except Exception as e:
                        print(e)
            # exit()
            for delta_d in delta_ds:
                for model_str in model_strs:
                    try:
                        plot_deltad(model_str, max_n, repeat, delta_d)
                    except Exception as e:
                        print(e)

        if sys.argv[1] == 's':
            for model_str in model_strs:
                for delta_d in delta_ds:
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
        delta_d = "0.00"
        filename = get_fname(model_str, 0, delta_d=delta_d) + "-s.txt"

        # for i in range(100):
        run_classes(filename, model_str, run_num=10, max_n=100, delta_d=delta_d)

        import plotly.express as px

        df = pd.read_csv(filename)
        # print(df.to_numpy())
        # fig = px.line(df)
        # fig.show()
