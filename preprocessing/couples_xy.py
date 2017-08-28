from sklearn.ensemble import IsolationForest
from sklearn.cluster import SpectralClustering
import numpy as np
from abc import ABC
import colors
from sklearn.base import BaseEstimator


class XyBase(ABC, BaseEstimator):
    def transform(self, data):
        if "couples_xy" in data:
            data["couples_xy"] = self.xy_transform(data["couples_xy"])
        return data

    def fit(self, *_):
        return self


class _SanitizeBase(XyBase):
    def __init__(self, contamination=.1):
        self.contamination = contamination

    def xy_transform(self, couples):
        iso = IsolationForest(contamination=self.contamination)

        # we want to turn the X, y format into a zipped list of starting and ending points
        zipped_couples = self.zipper(couples)

        # now make all dimensions have the same scale
        from sklearn.preprocessing import StandardScaler
        zipped_couples = StandardScaler().fit_transform(zipped_couples)

        iso.fit(zipped_couples)
        new_couples = [[], []]
        for i, label in enumerate(iso.predict(zipped_couples)):
            if label == 1:
                new_couples[0].append(couples[0][i])
                new_couples[1].append(couples[1][i])
        return new_couples

    def zipper(self, couples):
        pass


class SanitizeStartEnd(_SanitizeBase):
    def zipper(self, couples):
        return [np.concatenate([couples[0][i], couples[1][i]]) for i in range(len(couples[0]))]


class SanitizeStartVec(_SanitizeBase):
    def zipper(self, couples):
        return [x + (np.array(couples[1][i]) - np.array(x)).tolist() for i, x in enumerate(couples[0])]


class SanitizeVec(_SanitizeBase):
    def zipper(self, couples):
        return [(np.array(couples[1][i]) - np.array(x)).tolist() for i, x in enumerate(couples[0])]


def round_list(arr):
    return list(map(lambda x: round(x, 6), arr))


class Cluster(XyBase):
    # TODO: determine if pos/pos leads to the same classification as pos/vec
    # TODO: see if including position leads to an increase in accuracy, or if only the vector should be used
    def __init__(self, clusterer=None, replace=False):
        if clusterer is None:
            clusterer = SpectralClustering()
        self.clusterer = clusterer
        self.replace = replace

    def xy_transform(self, couples):
        try:
            dimensions = len(couples[0][0])

            # we again want to zip our lists, like in Sanitize
            zipped_couples = [round_list(x) + round_list(couples[1][i]) for i, x in enumerate(couples[0])]
            clusterer = self.clusterer.fit(zipped_couples)
            groups = {}
            for i, label in enumerate(clusterer.labels_):
                if not label in groups:
                    groups[label] = []
                groups[label].append(zipped_couples[i])

            # now we want to find the average of each group
            for group in groups:
                groups[group] = np.mean(groups[group], axis=0).tolist()

            # now we want to unzip our lists
            new_couples = [[], []]
            for group in groups:
                new_couples[0].append(groups[group][0:dimensions])
                new_couples[1].append(groups[group][dimensions:])

            if self.replace:
                return new_couples
            else:
                return [new_couples[0] + couples[0], new_couples[1] + couples[1]]

        except ValueError:
            # we have a smaller number of samples than desired clusters
            return couples

    def __repr__(self):
        return str(self.clusterer)
