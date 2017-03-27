from sklearn.ensemble import IsolationForest
from sklearn.cluster import SpectralClustering
import numpy as np


class Sanitize(object):
    def __init__(self, contamination=.1):
        self.contamination = contamination

    def transform(self, couples):
        iso = IsolationForest(contamination=self.contamination)

        # we want to turn the X, y format into a zipped list of starting and ending points
        zipped_couples = [x + couples[1][i] for i, x in enumerate(couples[0])]

        iso.fit(zipped_couples)
        new_couples = [[], []]
        for i, label in enumerate(iso.predict(zipped_couples)):
            if label == 1:
                new_couples[0].append(couples[0][i])
                new_couples[1].append(couples[1][i])
        return new_couples


class Cluster(object):
    def __init__(self, clusterer=None):
        if clusterer is None:
            clusterer = SpectralClustering()
        self.clusterer = clusterer

    def transform(self, couples):
        # we again want to zip our lists, like in Sanitize
        zipped_couples = [x + couples[1][i] for i, x in enumerate(couples[0])]

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
            new_couples[0].append(groups[group][0:5])
            new_couples[1].append(groups[group][5:10])
        return new_couples

    def __repr__(self):
        return self.clusterer