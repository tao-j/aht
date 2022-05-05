from sort import Sort


class MergeSort(Sort):
    def __init__(self, n):
        super().__init__()
        self.end = None
        self.right = None
        self.mid = None
        self.left = None
        self.nums = None
        self.temp = [0] * n
        self.temp_i = 0
        self.n = n

        self.sublist = 1
        self.dist = 2

        self.update_state()

        if self.n <= 1:
            self.done = True
        else:
            self.done = False

    def update_state(self):
        self.left = (self.sublist - 1) * self.dist
        self.mid = min(self.n, (self.sublist - 1) * self.dist + self.dist // 2)
        self.right = min(self.n, self.left + self.dist // 2)
        self.end = min(self.n, self.sublist * self.dist)

    def copy_left_leftover(self):
        if self.right == self.end:
            while self.left < self.mid:
                self.temp[self.temp_i] = self.nums[self.left]
                self.temp_i += 1
                self.left += 1

    def copy_right_leftover(self):
        if self.left == self.mid:
            while self.right < self.end:
                self.temp[self.temp_i] = self.nums[self.right]
                self.temp_i += 1
                self.right += 1

    def next_pair(self):
        if self.done:
            return [None, None]
        else:
            return [self.nums[self.left], self.nums[self.right]]

    def done(self):
        return self.done

    def feedback(self, y):
        if y != 1:  # left < right
            self.temp[self.temp_i] = self.nums[self.left]
            self.left += 1
        else:
            self.temp[self.temp_i] = self.nums[self.right]
            self.right += 1
        self.temp_i += 1

        # test if reaches boundary for sublist
        self.copy_right_leftover()
        self.copy_left_leftover()

        while self.right == self.end and self.left == self.mid and not self.done:
            if self.sublist * self.dist >= self.n:
                self.dist = self.dist * 2
                self.sublist = 1
                self.nums = self.temp
                self.temp = [None] * self.n
                self.temp_i = 0
            else:
                self.sublist += 1
            if self.dist // 2 > self.n:
                self.done = True
            else:
                self.update_state()
            # test if reaches boundary for this division
            self.copy_left_leftover()
        return

    def sort(self, array_in):
        self.nums = array_in
        while not self.done:
            pair = self.next_pair()
            if pair[0] > pair[1]:
                y = 1
            else:
                y = 0
            self.feedback(y)
        return self.nums


if __name__ == "__main__":
    import random
    import itertools

    random.seed(222)
    for n in range(0, 8):
        array_nums = list(range(100, 100 + n))
        for original_array in itertools.permutations(array_nums):
            mst = MergeSort(len(original_array))
            est_rank = mst.sort(original_array)
            a_mst = list(est_rank)
            print(a_mst)
            assert (a_mst == array_nums)
