class Sort:
    def __init__(self):
        self.done = False

    def done(self):
        return self.done

    def feedback(self, y):
        raise NotImplementedError

    def next_state(self):
        raise NotImplementedError

    def next_pair(self):
        raise NotImplementedError

    def sort(self, array_in) -> int:
        raise NotImplementedError

    def arg_sort(self):
        return NotImplementedError

    def request_pair(self, i, j):
        raise NotImplementedError
