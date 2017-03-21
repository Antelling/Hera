import numpy as np


class StdEqualizer(object):
    def transform(self, maps):
        # todo std equalizer
        return maps


class PosEqualizer(object):
    def transform(self, maps):
        # todo pos equalizer
        return maps


class MetricEqualizer(object):
    def __init__(self, metric=None):
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
            self.metric = metric[0]
            self.name = metric[1]

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
