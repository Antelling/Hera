import numpy as np

class ListBase(object):
    def transform(self, maps):
        maps["misc"][self.name] = self.form_list(maps["scoreable"][self.map])
        return maps


class OrderedRecs(ListBase):
    def __init__(self):
        self.name = "OrderedRecs"
        self.map = "main"

    def form_list(self, average_map):
        raw_recs = []
        for person in average_map:
            for otherperson in average_map[person]:
                raw_recs.append([person, otherperson[0], otherperson[1]])
        raw_recs.sort(key=lambda x: x[2])
        return raw_recs

class LeastCompatible(ListBase):
    def __init__(self):
        self.name = "LeastCompatible"
        self.map = "main"

    def form_list(self, average_map):
        people_map = {}
        for person in average_map:
            people_map[person] = []
        for person in average_map:
            for i, otherperson in enumerate(average_map[person]):
                people_map[otherperson[0]].append(i)
        compat_list = []
        for person in people_map:
            compat_list.append([person, np.mean(people_map[person])])
        compat_list.sort(key=lambda x:x[1])
        return compat_list

class BadCouples(ListBase):
    def __init__(self):
        self.name = "BadCouples"
        self.map = "main"

    def form_list(self, average_map):
        import data
        import preprocessing
        couples = data.get.couples_raw()
        couples = preprocessing.couples_raw.Mirror().transform(couples)
        from preprocessing.couples_raw import RANSAC
        bad_couples = RANSAC(max_iter=200, good=False).transform(couples)
        bad_couples = [str(couple["male"]) + " - " + str(couple["female"]) for couple in bad_couples]
        return bad_couples


