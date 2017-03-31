import math, copy, colors


class Time_mod(object):
    def __init__(self, mod=None):
        if mod is None:
            mod = lambda x: math.log(x)
            self.mod = mod
            self.name = "log"
        else:
            self.mod = mod[0]
            self.name = mod[1]

    def transform(self, couples):
        new_couples = []
        for couple in couples:
            time = int(self.mod(couple["length"]))
            for _ in range(time):
                c = copy.copy(couple)
                new_couples.append(c)
        return new_couples

    def __repr__(self):
        return self.name


class Mirror(object):
    def transform(self, couples):
        new_couples = copy.copy(couples)
        for couple in couples:
            c = copy.copy(couple)
            male = c["male"]
            c["male"] = c["female"]
            c["female"] = male
            new_couples.append(c)
        return new_couples
