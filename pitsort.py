import numpy as np


class Node:
    def __init__(self):
        self.parent = None
        self.left = None
        self.right = None
        self.mid = None
        self.lchild = None
        self.rchild = None
        self.count = 0


class PITSort:
    """
    ranks from low to high
    param N: number of items to rank
    
    returns argsort style index
    """
    def __init__(self, N, delta):
        self.N = N
        self.delta = delta

        if self.N == 0:
            self.done = True
            self.arg_list = []
            return
        if self.N == 1:
            self.done = True
            self.arg_list = [0]
            return
        else:
            self.done = False

        self.arg_list = [0]

        self.t_iir = 1
        self.t_iai = 1
        self.t_ati = 1
        self.delta_iai_param = self.delta / (self.N - 1)
        self.delta_ati_param = None
        self.delta_atc_param = None
        self.epsilon_ati_param = None
        self.epsilon_atc_param = None
        self.w = 0
        self.h = None
        self.prev_atc_y = None

        # N >= 2
        self.n_intree = 1
        self.requested_pair = [None, None]
        self.state = None
        self.X = None
        self.leaves = []
        self.rebuild_tree()
        self.next_state()
        self.debug_b = 1

    def rebuild_tree(self):
        root = Node()
        self.root = root
        self.X = root
        self.leaves = []
        root.left = -1
        root.right = self.n_intree
        mid = (root.left + root.right) // 2
        root.mid = mid

        bq = []
        bq.append(root)
        # build the index/interval tree
        while len(bq):
            current = bq[0]
            bq.pop(0)
            mid = (current.left + current.right) // 2
            current.mid = mid
            if current.right - current.left > 1:
                ltemp = Node()
                rtemp = Node()
                ltemp.parent = current
                ltemp.left = current.left
                ltemp.right = mid
                current.lchild = ltemp
                bq.append(ltemp)

                rtemp.parent = current
                rtemp.left = mid
                rtemp.right = current.right
                current.rchild = rtemp
                bq.append(rtemp)
            else:
                self.leaves.append(current)

    def next_state(self):
        q = 15 / 16
        q2 = np.sqrt(q)
        q3 = np.power(q, 1 / 3)
        if self.state == 1:
            self.state = 2
            self.delta_atc_param = 1 - q2
            self.requested_pair = self.n_intree, self.X.right
        elif self.state == 3:
            self.state = 4
            self.delta_atc_param = 1 - q3
            self.requested_pair = self.n_intree, self.X.right
        elif self.state == 5:
            self.delta_atc_param = 1 - q3
            self.requested_pair = self.n_intree, self.X.mid
        # root node
        elif self.X.left == -1 and self.X.right == self.n_intree:
            self.state = 0
            self.delta_atc_param = 1 - q
            self.requested_pair = self.n_intree, self.X.mid
        # leaf node
        elif self.X.right - self.X.left == 1:
            self.state = 1
            self.delta_atc_param = 1 - q2
            self.requested_pair = self.n_intree, self.X.left
        else:
            self.state = 3
            self.delta_atc_param = 1 - q3
            self.requested_pair = self.n_intree, self.X.left

        self.delta_ati_param = 6 * self.delta_iai_param / np.pi / np.pi / self.t_iai / self.t_iai
        self.epsilon_ati_param = np.power(2., -(self.t_iai + 1))
        self.epsilon_atc_param = self.epsilon_ati_param
        self.h = np.ceil(1 + np.log2(1 + self.n_intree))

    def next_pair(self):
        return self.requested_pair

    def done(self):
        return self.done

    # if y == 1 then request(a, b) returns a > b
    def feedback(self, atc_y, total = 0):
        inserted = False
        inserted_place = -1

        t = self.t_ati
        t_max = np.ceil(np.max([4 * self.h, 512 / 25 * np.log2(2 / self.delta_ati_param)]))

        # state number means I have already reached number of ATC calls in this iteration
        if self.state == 0:
            self.X = self.X.rchild if atc_y == 1 else self.X.lchild
            assert self.X is not None
        elif self.state == 1:
            assert self.X.parent is not None
            self.prev_atc_y = atc_y
            self.t_ati -= 1
        elif self.state == 2:
            assert self.X.parent is not None
            if self.prev_atc_y and atc_y == 0:
                self.X.count += 1
                b_t = 0.5 * t + np.sqrt(t / 2 * np.log2(np.pi * np.pi * t * t / 3 / self.delta_ati_param)) + 1
                if self.X.count > b_t:
                    inserted = True
                    self.arg_list.insert(self.X.right, self.n_intree)
                    inserted_place = self.X.right
                    self.n_intree += 1
            elif self.X.count > 0:
                self.X.count -= 1
            else:
                self.X = self.X.parent
                assert self.X is not None
        elif self.state == 3:
            self.prev_atc_y = atc_y
            self.t_ati -= 1
        elif self.state == 4:
            if self.prev_atc_y == 0 or atc_y:
                self.X = self.X.parent
                assert self.X is not None
            else:
                self.state = 5
                self.t_ati -= 1
        elif self.state == 5:
            self.X = self.X.rchild if atc_y == 1 else self.X.lchild
            self.state = 6
            assert self.X is not None

        if inserted or self.t_ati == t_max:
            if not inserted:
                for node in self.leaves:
                    if node.count > 1 + 5 / 16 * t_max:
                        inserted = True
                        inserted_place = node.right
                        self.arg_list.insert(node.right, self.n_intree)
                        self.n_intree += 1
                        break
            if inserted:
                self.t_iai = 1
                self.t_iir += 1
                if total != 0:
                    print(total)
                if self.t_iir == self.N:
                    self.done = 1
                    return inserted, inserted_place
            else:
                self.t_iai += 1
            self.t_ati = 1
            self.rebuild_tree()
        else:
            self.t_ati += 1

        self.next_state()
        return inserted, inserted_place


def cmp(i1, i2, arglist, original_a):
    return 1 if original_a[i1] > original_a[arglist[i2]] else 0


if __name__ == "__main__":
    import random
    import itertools

    random.seed(222)
    for n in range(1, 30):
        rand_nums = []
        for i in range(n):
            rand_nums.append(random.randint(1, 100))
        for a in itertools.permutations(rand_nums):
            a_sorted = sorted(a)
            # print("as: ", a_sorted)
            cmp_sort = PITSort(len(a), 0.1)
            while not cmp_sort.done:
                pair = cmp_sort.next_pair()
                assert (0 <= pair[0] <= cmp_sort.n_intree)
                assert (-1 <= pair[1] <= cmp_sort.n_intree)
                if pair[1] == -1:
                    cmp_sort.feedback(1)
                elif pair[1] == cmp_sort.n_intree:
                    cmp_sort.feedback(0)
                else:
                    y = cmp(pair[0], pair[1], cmp_sort.arg_list, a)
                    cmp_sort.feedback(y)
                # print("ag: ", cmp_sort.arg_list)
            a_ps = [0] * n
            for idx, i in enumerate(cmp_sort.arg_list):
                a_ps[idx] = a[i]
            a_ps = list(np.array(a)[cmp_sort.arg_list])
            # print("ps: ", cmp_sort.arg_list)
            # print("aps:", a_ps)
            assert (a_ps == a_sorted)
            break
