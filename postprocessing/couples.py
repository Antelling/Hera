import data

class JVCouples(object):
    def __init__(self, loss_function=None):
        if loss_function is None:
            loss_function = lambda x: x
        self.loss = loss_function

    def transform(self, maps):
        maps["misc"]["JVCouples"] = self.form_couples(maps["scoreable"]["average"])
        return maps

    def form_couples(self, maps):
        from lapjv import lapjv

        people = data.get.people_raw()

        men = []
        women = []
        for person in maps:
            if people[person]["gender"] == "male":
                men.append(person)
            else:
                women.append(person)

        # we need to make men and women the same length
        difference = len(men) - len(women)
        if difference == 0:
            pass
        elif difference < 0:
            # there are more women then men
            women = self.remove_people(maps, women, difference * -1)
        else:
            men = self.remove_people(maps, men, difference)

        #now we need to make our cost matrix
        #for every man, append a list to the cost matrix, containing the male distance to the female

        sane_map = {}
        for person in maps:
            sane_map[person] = {}
            for otherperson in maps[person]:
                sane_map[person][otherperson[0]] = otherperson[1]

        cost_matrix = []
        for man in men:
            costs = []
            for woman in women:
                costs.append(sane_map[man][woman])
            cost_matrix.append(costs)

        pairs = lapjv(cost_matrix)

        couples = {}
        for i, target in enumerate(pairs[0]):
            couples[men[i]] = women[target]
            couples[women[target]] = men[i]

        return couples

    def remove_people(self, maps, people, n):
        # we need to sort people to find who has the highest average place
        # we already implemented this as the get_summary_stats method of MetricEqualizer, so let's just steal that
        from .normalize import MetricEqualizer
        stats_func = MetricEqualizer().get_summary_stats
        people_stats = stats_func(maps)
        people.sort(key=lambda name: people_stats[name][0])  # sort by mean
        return people[0:len(people) - n]
