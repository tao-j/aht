from sort import Sort

import numpy as np


class Node:
    def __init__(self):
        self.parent = None
        self.left = None
        self.right = None
        self.mid = None
        self.l_child = None
        self.r_child = None
        self.count = 0


class PITSort(Sort):
    """
    ranks from low to high
    param N: number of items to rank
    
    returns argsort style index
    """

    def __init__(self, n, delta, model):
        self.N = n
        self.delta_rank = delta
        self.model = model
        super(PITSort, self).__init__()

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
        self.delta_iai_param = self.delta_rank / (self.N - 1)
        self.delta_ati_param = None
        self.delta_atc_param = None
        self.epsilon_ati_param = None
        self.epsilon_atc_param = None
        self.w = 0
        self.h = None
        self.prev_atc_y = None

        # N >= 2
        self.n_in_tree = 1
        self.requested_pair = [-1, -1]
        self.state = None
        self.X = None
        self.leaves = []
        self.rebuild_tree()
        self.next_state()
        self.debug_b = 1

        root = Node()
        self.root = root

        self.sample_complexity = 0

    def rebuild_tree(self):
        root = Node()
        self.root = root
        self.X = root
        self.leaves = []
        root.left = -1
        root.right = self.n_in_tree
        mid = (root.left + root.right) // 2
        root.mid = mid

        bq = [root]
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
                current.l_child = ltemp
                bq.append(ltemp)

                rtemp.parent = current
                rtemp.left = mid
                rtemp.right = current.right
                current.r_child = rtemp
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
            self.requested_pair = self.n_in_tree, self.X.right
        elif self.state == 3:
            self.state = 4
            self.delta_atc_param = 1 - q3
            self.requested_pair = self.n_in_tree, self.X.right
        elif self.state == 5:
            self.delta_atc_param = 1 - q3
            self.requested_pair = self.n_in_tree, self.X.mid
        # root node
        elif self.X.left == -1 and self.X.right == self.n_in_tree:
            self.state = 0
            self.delta_atc_param = 1 - q
            self.requested_pair = self.n_in_tree, self.X.mid
        # leaf node
        elif self.X.right - self.X.left == 1:
            self.state = 1
            self.delta_atc_param = 1 - q2
            self.requested_pair = self.n_in_tree, self.X.left
        else:
            self.state = 3
            self.delta_atc_param = 1 - q3
            self.requested_pair = self.n_in_tree, self.X.left

        self.delta_ati_param = 6 * self.delta_iai_param / np.pi / np.pi / self.t_iai / self.t_iai
        self.epsilon_ati_param = np.power(2., -(self.t_iai + 1))
        self.epsilon_atc_param = self.epsilon_ati_param
        self.h = np.ceil(1 + np.log2(1 + self.n_in_tree))

    def next_pair(self):
        return self.requested_pair

    # if y == 1 then request(a, b) returns a > b
    def feedback(self, atc_y, total=0):
        inserted = False
        inserted_place = -1

        t = self.t_ati
        t_max = np.ceil(np.max([4 * self.h, 512 / 25 * np.log2(2 / self.delta_ati_param)]))

        # state number equals the number of ATC calls minus 1 reached in this iteration
        # X is root
        if self.state == 0:
            self.X = self.X.r_child if atc_y == 1 else self.X.l_child
            assert self.X is not None
        # X is leaf
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
                    self.arg_list.insert(self.X.right, self.n_in_tree)
                    inserted_place = self.X.right
                    self.n_in_tree += 1
            elif self.X.count > 0:
                self.X.count -= 1
            else:
                self.X = self.X.parent
                assert self.X is not None
        # X is node
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
            self.X = self.X.r_child if atc_y == 1 else self.X.l_child
            self.state = 6
            assert self.X is not None

        if inserted or self.t_ati == t_max:
            if not inserted:
                for node in self.leaves:
                    if node.count > 1 + 5 / 16 * t_max:
                        inserted = True
                        inserted_place = node.right
                        self.arg_list.insert(node.right, self.n_in_tree)
                        self.n_in_tree += 1
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

    def atc(self, i, j, eps, delta):
        t_max = int(np.ceil(1. / 2 / (eps ** 2) * np.log2(2 / delta)))
        p = 0.5
        w = 0
        t = np.arange(1, t_max + 1)
        bb_t = np.sqrt(1. / 2 / t * np.log2(np.pi * np.pi * t * t / 3 / delta))
        for t in range(1, t_max + 1):
            y = self.request_pair(i, j)
            assert y == 1 or y == 0
            if y == 1:
                w += 1
            b_t = bb_t[t - 1]
            p = w / t
            if p > 0.5 + b_t:
                break
            if p < 0.5 - b_t:
                break

        self.sample_complexity += t
        atc_y = 1 if p > 0.5 else 0
        return atc_y

    def post_atc(self, inserted, inserted_place):
        pass

    def request_pair(self, i, j):
        y = self.model.sample_pair(i, j)
        return y

    def arg_sort(self):
        while not self.done:
            pair = self.next_pair()
            assert (0 <= pair[0] <= self.n_in_tree)
            assert (-1 <= pair[1] <= self.n_in_tree)
            if pair[1] == -1:
                inserted, inserted_place = self.feedback(1)
            elif pair[1] == self.n_in_tree:
                inserted, inserted_place = self.feedback(0)
            else:
                y = self.atc(pair[0], self.arg_list[pair[1]],
                             self.epsilon_atc_param, self.delta_atc_param)
                inserted, inserted_place = self.feedback(y)
                # y = model.sample_pair(pair[0], self.arg_list[pair[1]])
                # inserted, _ = self.feedback(y)
                # if not inserted:
                #     i = pair[0]
                #     j = self.arg_list[pair[1]]
                #     print(pair, y, model.Pij[0, i, j])
            self.post_atc(inserted, inserted_place)
            # if inserted:
            #     print("inserted idx:", pair[0], self.arg_list, "real:", np.array(self.model.s)[self.arg_list])
        return self.arg_list


if __name__ == "__main__":
    import random
    from models import DummyModel

    random.seed(222)
    for n in range(0, 16):
        array_nums = np.arange(100, 100 + n)
        for _ in range(min(n + 1, n * n + 1)):
            array_original = np.random.permutation(array_nums)
            a_sorted = list(np.argsort(array_original))
            print("arg: ", a_sorted)
            d_model = DummyModel(array_original)
            pit_s = PITSort(len(array_original), 0.1, d_model)
            pit_a = pit_s.arg_sort()
            print("pit: ", pit_a)
            assert (pit_a == a_sorted)
