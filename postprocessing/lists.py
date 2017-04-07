import numpy as np

class ListBase(object):
    def __init__(self, name="list"):
        self.name = name
        self.map = "main"

    def transform(self, maps):
        maps["misc"][self.name] = self.form_list(maps["scoreable"][self.map])
        return maps


class OrderedRecs(ListBase):
    def form_list(self, main_map):
        raw_recs = []
        for person in main_map:
            for otherperson in main_map[person]:
                raw_recs.append([person, otherperson[0], otherperson[1]])
        raw_recs.sort(key=lambda x: x[2])
        return raw_recs

class LeastCompatible(ListBase):
    def form_list(self, main_map):
        people_map = {}
        for person in main_map:
            people_map[person] = []
        for person in main_map:
            for i, otherperson in enumerate(main_map[person]):
                people_map[otherperson[0]].append(i)
        compat_list = []
        for person in people_map:
            compat_list.append([person, np.mean(people_map[person])])
        compat_list.sort(key=lambda x:x[1])
        return compat_list

class RansacBadCouples(ListBase):
    def form_list(self, main_map):
        import data
        import preprocessing
        couples = data.get.couples_raw()
        couples = preprocessing.couples_raw.Mirror().transform(couples)
        from preprocessing.couples_raw import RANSAC
        bad_couples = RANSAC(max_iter=200, good=False).transform(couples)
        bad_couples = [str(couple["male"]) + " - " + str(couple["female"]) for couple in bad_couples]
        return bad_couples

class RedBadCouples(ListBase):
    def form_list(self, main_map, couples=None):
        if couples is None:
            import data, preprocessing
            couples = data.get.couples_raw()
            couples = preprocessing.couples_raw.Mirror().transform(couples)
        #we want to remove couples that are in the bottom of each other's recs
        bad_couples = []
        for couple in couples:
            source = couple["male"]
            target = couple["female"]
            rec_list = [otherperson[0] for otherperson in main_map[source]]
            length = len(rec_list)
            index = rec_list.index(target)
            max = length * .8
            if index > max:
                bad_couples.append(couple["male"].split(" ")[0] + " -> " + couple["female"].split(' ')[0])
        return bad_couples