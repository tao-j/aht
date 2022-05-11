from pitsort import PITSort
from probesort import ProbeSortUC, ProbeSortULC, ProbeSortUT, ProbeSortULT
from models import WSTModel, HBTL

import numpy as np
import random
import plotly.express as px
import pandas as pd


def test_classes(n, class_list):
    delta = 0.25

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


def run_classes(filename):
    random.seed(333)
    np.random.seed(333)
    fout = open(filename, 'w')
    classes_to_test = [PITSort, ProbeSortUC, ProbeSortULC, ProbeSortUT, ProbeSortULT]
    fout.write(",".join(map(lambda cls: cls.__name__, classes_to_test)) + "\n")
    for i in range(10, 100, 10):
        res = test_classes(i, classes_to_test)
        fout.write("{}\n".format(",".join(res)))
        fout.flush()
    fout.close()


if __name__ == "__main__":
    filename = "output/pbs-hbtl-cmp4.txt"
    run_classes(filename)
    df = pd.read_csv(filename)
    # fig = px.histogram(df)
    fig = px.line(df)
    fig.show()
