from pitsort import PITSort
from probesort import ProbeSort
from models import WSTModel, HBTL

import numpy as np
import random
import plotly.express as px
import pandas as pd


def test():
    n = 20
    delta = 0.1

    init_rank = list(np.arange(n))

    gt_rank = list(np.random.permutation(n))
    model = WSTModel(gt_rank, slackness=0.01)
    # s = np.random.rand(n)
    # gamma = np.ones(1)
    # model = HBTL(s, gamma)
    # gt_rank = list(np.argsort(s))

    pit_s = PITSort(n, delta)
    prb_s = ProbeSort(n, delta)

    print(gt_rank, "gt")
    print(init_rank, "init")

    rank1 = pit_s.sort(init_rank, model)
    s1 = pit_s.sample_complexity
    rank1.reverse()
    print(rank1, s1)

    rank2 = prb_s.sort(None, model)
    s2 = prb_s.sample_complexity
    print(rank2, s2)

    return [s1, s2]


filename = "data/pbs2.txt"


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
