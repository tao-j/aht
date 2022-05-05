from pitsort import PITSort
from probesort import ProbeSort
from models import WSTModel, HBTL

import numpy as np
import random
import plotly.express as px
import pandas as pd


def test():
    n = 5
    delta = 0.1

    # gt_a = list(np.random.permutation(n))
    # model = WSTModel(gt_a, slackness=0.01)
    s = np.random.rand(n)
    gamma = np.ones(1)
    model = HBTL(s, gamma)
    gt_a = list(np.argsort(s))

    pit_s = PITSort(n, delta, model)
    prb_s = ProbeSort(n, delta, model)

    print(gt_a, "gt")

    pit_a = pit_s.arg_sort()
    s1 = pit_s.sample_complexity
    print(pit_a, s1)

    prb_a = prb_s.arg_sort()
    s2 = prb_s.sample_complexity
    print(prb_a, s2)

    assert gt_a == prb_a
    assert gt_a == pit_a
    return [s1, s2]


filename = "output/pbs2.txt"


def run():
    random.seed(333)
    np.random.seed(333)
    res = []
    fout = open(filename, 'w')
    fout.write("pit,pbs\n")
    for i in range(100):
        s1, s2 = test()
        fout.write("{},{}\n".format(s1, s2))
    fout.close()


if __name__ == "__main__":
    run()
    df = pd.read_csv(filename)
    fig = px.histogram(df)
    fig.show()
