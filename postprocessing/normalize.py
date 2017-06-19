import numpy as np
import data, colors
import json


class MetricEqualizer(object):
    # TODO: add position as a metric, to emulate position equalizer
    def __init__(self, metric=None, name=None):
        if metric is None:
            metric = "zscore"
        if metric == "zscore":
            self.metric = lambda x, mean, median, std: (x - mean) / std
            self.name = "ZScoreNormalized"
        elif metric == "percentage":
            self.metric = lambda x, mean, median, std: x / mean
            self.name = "PercentageNormalized"
        elif metric == "zscore_median":
            self.metric = lambda x, mean, median, std: (x - median) / std
            self.name = "ZScoreMedianNormalized"
        elif metric == "percentage_median":
            self.metric = lambda x, mean, median, std: x / median
            self.name = "PercentageMedianNormalized"
        elif metric == "distance":
            self.metric = lambda x, mean, median, std: x - mean
            self.name = "DistanceNormalized"
        elif metric == "distance_median":
            self.metric = lambda x, mean, median, std: x - median
            self.name = "DistanceMedianNormalized"
        else:
            self.metric = metric
            self.name = "CustomMetric"

        if name is not None:
            self.name = name

    def transform(self, maps):
        map_to_use = "average" if "average" in maps["scoreable"] else "one-way"
        maps["scoreable"][self.name] = self.normalize_map(maps["scoreable"][map_to_use])
        return maps

    def normalize_map(self, ave_map):
        # for every person, we need their average distance
        dist_normal = {}
        summary_stats = self.get_summary_stats(ave_map)

        for person in ave_map:
            dist_list = []
            for otherperson in ave_map[person]:
                metric_score = self.metric(otherperson[1], *summary_stats[otherperson[0]])
                dist_list.append([otherperson[0], metric_score])
            dist_list.sort(key=lambda x: x[1])
            dist_normal[person] = dist_list

        return dist_normal

    def get_summary_stats(self, best_couples):
        """finds the average distance each person is"""
        names = {}
        for name in best_couples:
            names[name] = []
        for _ in best_couples:
            for person in best_couples[_]:
                names[person[0]].append(person[1])
        for name in names:
            names[name] = [np.mean(names[name]), np.median(names[name]), np.std(names[name])]
        return names


class CoupleEqualizer(object):
    """Uses linear assignment between all genders to normalize recommendations"""
    def __init__(self, name="CouplesNormalized"):
        self.name = name

    def transform(self, maps):
        maps["scoreable"][self.name] = self.form_map(maps["scoreable"]["average"])
        return maps

    def form_map(self, maps):
        colors.blue("starting...")
        from sklearn.utils.linear_assignment_ import linear_assignment

        people_list = []

        sane_map = {}
        for person in maps:
            sane_map[person] = {}
            for otherperson in maps[person]:
                sane_map[person][otherperson[0]] = otherperson[1]

        for person in maps:
            people_list.append(person)

        finished_map = {}
        for person in people_list:
            finished_map[person] = []
        for i in range(len(people_list)):
            cost_matrix = []
            for person in people_list:
                costs = []
                for otherperson in people_list:
                    costs.append(sane_map[person][otherperson])
                cost_matrix.append(costs)

            pairs = linear_assignment(np.array(cost_matrix))
            print(str(i) + "/" + str(len(people_list)))

            for couple in enumerate(pairs):
                couple = couple[1]
                male_name = people_list[couple[0]]
                female_name = people_list[couple[1]]
                finished_map[male_name].append([female_name, sane_map[male_name][female_name]])
                sane_map[male_name][female_name] = 1000 + i

        colors.green("done")
        return finished_map


class CoupleEqualizerFast(object):
    """Uses linear assignment between opposite genders to normalize recommendations"""
    def __init__(self, name="CouplesNormalized"):
        self.name = name

    def transform(self, maps):
        maps["scoreable"][self.name] = self.form_map(maps["scoreable"]["average"])
        return maps

    def form_map(self, maps):
        from sklearn.utils.linear_assignment_ import linear_assignment
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
            removed = []
        elif difference < 0:
            # there are more women then men
            women, removed = self.remove_people(maps, women, difference * -1)
        else:
            men, removed = self.remove_people(maps, men, difference)

        sane_map = {}
        for person in maps:
            sane_map[person] = {}
            for otherperson in maps[person]:
                sane_map[person][otherperson[0]] = otherperson[1]

        finished_map = {}
        for man in men:
            finished_map[man] = []
        for woman in women:
            finished_map[woman] = []
        for i in range(len(men)):
            cost_matrix = []
            for man in men:
                costs = []
                for woman in women:
                    costs.append(sane_map[man][woman])
                cost_matrix.append(costs)

            pairs = linear_assignment(np.array(cost_matrix))

            for couple in enumerate(pairs):
                couple = couple[1]
                male_name = men[couple[0]]
                female_name = women[couple[1]]
                sane_map[male_name][female_name] = 999999 + i
                sane_map[female_name][male_name] = 999999 + i
                finished_map[male_name].append([female_name, i])
                finished_map[female_name].append([male_name, i])

        for person in removed:
            finished_map[person] = []
            for otherperson in maps[person]:
                finished_map[person].append([otherperson[0], otherperson[1] + 1000])
        #okay so now the removed people are all matched with someone
        #we need to match everyone else with the removed people
        for person in finished_map:
            if person in removed:
                continue
            for removed_person in removed:
                if removed_person == person:
                    continue
                finished_map[person].append([removed_person, 9999])


        return finished_map

    def remove_people(self, maps, people, n):
        # we need to sort people to find who has the highest average place
        # we already implemented this as the get_summary_stats method of MetricEqualizer, so let's just steal that
        from .normalize import MetricEqualizer
        stats_func = MetricEqualizer().get_summary_stats
        people_stats = stats_func(maps)
        people.sort(key=lambda name: people_stats[name][0])  # sort by mean
        return people[0:len(people) - n], people[len(people) - n:]