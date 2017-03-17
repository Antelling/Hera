class filter_outliers(object):
    def __init__(self, contamination=.1):
        self.contamination = contamination

    def transform(self, couples):
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=self.contamination)
        # TODO: remove couples that are labeled as outliers by the isolation forest
        return couples


class cluster(object):
    def __init__(self, clusterer):
        self.clusterer = clusterer

    def transform(self, couples):
        # TODO: cluster couples
        # TODO: average couple groups
        return couples
