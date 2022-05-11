from pitsort import PITSort
from probesort import ProbeSort, ProbeSortA
from models import WSTModel, HBTL

import numpy as np
import random
import plotly.express as px
import pandas as pd


def test(n):
    # n = 5
    delta = 0.1

    gt_a = list(np.random.permutation(n))
    model = WSTModel(gt_a, slackness=0.25)
    
    # s = np.arange(1, n + 1)
    # gamma = np.ones(1)
    #model = HBTL(s, gamma)
    # gt_a = list(np.argsort(s))

    pit_s = PITSort(n, delta, model)
    prbo_s = ProbeSort(n, delta, model)
    prba_s = ProbeSortA(n, delta, model)

    print(gt_a, "gt")

    pit_a = pit_s.arg_sort()
    s1 = pit_s.sample_complexity
    print("pit_a ", s1)

    prbo_a = prbo_s.arg_sort()
    s2 = prbo_s.sample_complexity
    print("prbo_a", s2)

    prba_a = prba_s.arg_sort()
    s3 = prba_s.sample_complexity
    print("prba_a", s3)

    assert gt_a == prbo_a
    assert gt_a == prba_a
    assert gt_a == pit_a
    return [s1, s2, s3]


filename = "output/pbs-wst3.txt"


def run():
    random.seed(333)
    np.random.seed(333)
    res = []
    fout = open(filename, 'w')
    fout.write("pit,pbso,pbsa\n")
    for i in range(10, 100, 10):
        s1, s2, s3 = test(i)
        fout.write("{},{},{}\n".format(s1, s2, s3))
        fout.flush()
    fout.close()


if __name__ == "__main__":
    run()
    df = pd.read_csv(filename)
    # fig = px.histogram(df)
    fig = px.line(df)
    fig.show()
