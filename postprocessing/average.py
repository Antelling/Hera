class Average(object):
    """Makes loss between people symmetric"""

    def __init__(self, name="average"):
        self.name = name

    def transform(self, maps):
        maps["scoreable"][self.name] = self.average_map(maps["scoreable"]["one-way"])
        return maps

    def average_map(self, maps):
        new_map = {}
        for person in maps:
            new_map[person] = {}
            for otherperson in maps[person]:
                new_map[person][otherperson[0]] = otherperson[1]
        averaged_map = {}
        for person in new_map:
            if not person in averaged_map:
                averaged_map[person] = {}
            for otherperson in new_map[person]:
                if not otherperson in averaged_map:
                    averaged_map[otherperson] = {}
                a = new_map[person][otherperson]
                b = new_map[otherperson]
                b = b[person]
                averaged_dist = (a + b) / 2
                averaged_map[person][otherperson] = averaged_dist
                averaged_map[otherperson][person] = averaged_dist
        list_map = {}
        for person in averaged_map:
            list_map[person] = []
            for otherperson in averaged_map[person]:
                list_map[person].append([otherperson, averaged_map[person][otherperson]])
            list_map[person].sort(key=lambda x: x[1])
        return list_map
