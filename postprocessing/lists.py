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
