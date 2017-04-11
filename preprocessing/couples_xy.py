from sklearn.ensemble import IsolationForest
from sklearn.cluster import SpectralClustering
import numpy as np


class _SanitizeBase(object):
    def __init__(self, contamination=.1):
        self.contamination = contamination

    def transform(self, couples):
        iso = IsolationForest(contamination=self.contamination)

        # we want to turn the X, y format into a zipped list of starting and ending points
        zipped_couples = self.zipper(couples)

        #now make all dimensions have the same scale
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
        return [x + couples[1][i] for i, x in enumerate(couples[0])]

class SanitizeStartVec(_SanitizeBase):
    def zipper(self, couples):
        return [x + (np.array(couples[1][i]) - np.array(x)).tolist() for i, x in enumerate(couples[0])]

class SanitizeVec(_SanitizeBase):
    def zipper(self, couples):
        return [(np.array(couples[1][i]) - np.array(x)).tolist() for i, x in enumerate(couples[0])]


def round_list(arr):
    return list(map(lambda x:round(x, 6), arr))

class Cluster(object):
    # TODO: determine if pos/pos leads to the same classification as pos/vec
    # TODO: see if including position leads to an increase in accuracy, or if only the vector should be used
    def __init__(self, clusterer=None):
        if clusterer is None:
            clusterer = SpectralClustering()
        self.clusterer = clusterer

    def transform(self, couples):
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
        return new_couples

    def __repr__(self):
        return str(self.clusterer)
