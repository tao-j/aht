import os
import pandas as pd
import numpy as np
import plotly.express as px

base_dir = "data/Crowdflower_Comparisons"

ev_dir, ic_dir, pa_dir, pn_dir = "EventTime  ImageClarity  PeopleAge  PeopleNum".split(
    "  ")
out_dir = "data"


def pn():

    p = open(os.path.join(base_dir, pn_dir, "mall.txt"))
    out_f = open(os.path.join(out_dir, "pn.txt"), 'w')

    first_line = True

    idmap = {}
    last_new_id = 1

    for line in p:
        if first_line:
            first_line = False
        else:
            sline = line.strip().split(',')
            url1 = sline[-2]
            url2 = sline[-1]
            worker_id = sline[9]
            if worker_id not in idmap:
                idmap[worker_id] = last_new_id
                last_new_id += 1
            worker_id = idmap[worker_id]
            xurl1 = int(url1.split('mall_dataset/')[1].split('.')[0])
            xurl2 = int(url2.split('mall_dataset/')[1].split('.')[0])
            category = sline[-5]

            if category == 'category1':
                out_f.write("{} {} {}\n".format(worker_id, xurl1 + 1,
                                                xurl2 + 1))

            elif category == 'category2':
                out_f.write("{} {} {}\n".format(worker_id, xurl2 + 1,
                                                xurl1 + 1))


def ev():
    p = open(os.path.join(base_dir, ev_dir, "events.txt"))
    out_f = open(os.path.join(out_dir, "ev.txt"), 'w')
    first_line = True

    idmap = {}
    last_new_id = 1
    for line in p:
        sline = line.strip().split(',')
        if first_line == True:
            first_line = False

        else:
            try:
                first_node = int(sline[-2])
                second_node = int(sline[-1])
                tainted = sline[6]
                worker_id = sline[9]
                category = sline[-7]
                if tainted != 'false':
                    continue
                
                if worker_id not in idmap:
                    idmap[worker_id] = last_new_id
                    last_new_id += 1
                worker_id = idmap[worker_id]
                if category[-1] == '1':
                    out_f.write("{} {} {}\n".format(worker_id, first_node + 1,
                                                    second_node + 1))
                else:
                    out_f.write("{} {} {}\n".format(worker_id, second_node + 1,
                                                    first_node + 1))
            except Exception as e:
                # print(e)
                pass

# id, win, lose
# pn()
ev()
sn = "ev"
# %%
dat = np.loadtxt(f"data/{sn}.txt", dtype=np.int32)
correct = dat[:, 1] < dat[:, 2]
print("max id", np.max(np.max(dat[:, 1:])))
print("min id", np.min(np.min(dat[:, 1:])))
# %%

np.sum(correct) / len(dat)
n = np.max(dat[:, 0])
n_correct = np.zeros(n)
n_ans = np.zeros(n)
for i in range(len(dat)):
    k = dat[i, 0] - 1
    if dat[i, 1] < dat[i, 2]:
        n_correct[k] += 1
    n_ans[k] += 1
mu = n_correct * 1. / n_ans
print(np.mean(mu), np.std(mu), "mean std")
sel = mu[n_ans > 100]
# %%

sel = mu[n_ans > 100]
print(len(sel))
np.savetxt(f"{sn}-acc100.txt", sel)
sel = mu[n_ans > 200]
np.savetxt(f"{sn}-acc200.txt", sel)
print(len(sel))
fig = px.histogram(sel, nbins=64)
fig.show()
sel = mu[n_ans > 300]
np.savetxt(f"{sn}-acc300.txt", sel)
print(len(sel))
