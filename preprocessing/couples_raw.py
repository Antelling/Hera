import math


class time_mod(object):
    def __init__(self, mod=None):
        if mod is None:
            mod = lambda x: math.log(x)
            self.mod = mod

    def transform(self, couples):
        # TODO: replicate couples according to self.mod(couple.time)
        return couples


class mirror(object):
    def __init__(self):
        pass

    def transform(self, couples):
        # TODO: replicate every couple about gender
        return couples
