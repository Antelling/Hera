import math, copy, random
import colors, data
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class Time_mod(object):
    def __init__(self, mod=None):
        if mod is None:
            mod = lambda x: math.log(x)
            self.mod = mod
            self.name = "log"
        else:
            self.mod = mod[0]
            self.name = mod[1]

    def transform(self, couples):
        new_couples = []
        for couple in couples:
            time = int(self.mod(couple["length"]))
            for _ in range(time):
                c = copy.copy(couple)
                new_couples.append(c)
        return new_couples

    def __repr__(self):
        return self.name


class Mirror(object):
    def transform(self, couples):
        new_couples = copy.copy(couples)
        for couple in couples:
            c = copy.copy(couple)
            male = c["male"]
            c["male"] = c["female"]
            c["female"] = male
            new_couples.append(c)
        return new_couples


class RANSAC(object):
    def __init__(self, max_iter=100, good=True, min_consensus=999999):
        self.max_iter = max_iter
        self.min_consensus=min_consensus
        self.good = good

    def transform(self, couples):
        X, y = data.make.couples_xy(couples)
        from sklearn.linear_model import LinearRegression
        from wrappers import SklearnWrapper
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        ran_results = self.ran(SklearnWrapper(make_pipeline(PolynomialFeatures(3), LinearRegression()), unidirectional=True), [X, y], couples)
        if self.good:
            return ran_results["good_couples"]
        else:
            return ran_results["bad_couples"]

    def ran(self, model, Xy, couples):
        threshold = self.mad(Xy[0])

        best = {"consensus": 0}
        for _ in range(self.max_iter):
            encoded = self.encode(*Xy)
            random.shuffle(encoded)
            possible_inliers = encoded[0:random.randint(2, len(encoded) - 5)]
            model.fit(*self.decode(possible_inliers))
            predictions = model.predict(Xy[0])
            residuals = self.calc_residuals(list(map(lambda x: x[0], predictions)), Xy[1])
            inliers = 0
            good_couples = []
            bad_couples = []
            for i, resid in enumerate(residuals):
                if resid < threshold:
                    inliers += 1
                    good_couples.append(couples[i])
                else:
                    bad_couples.append(couples[i])
            if inliers > best["consensus"]:
                best["consensus"] = len(possible_inliers)
                best["good_couples"] = good_couples
                best["bad_couples"] = bad_couples
            if best["consensus"] > self.min_consensus:
                break

        return best

    def mad(self, arr):
        # like std but based on median so its robust
        arr = np.ma.array(arr).compressed()
        med = np.median(arr)
        return np.median(np.abs(arr - med))

    def encode(self, arr1, arr2):
        self.pivot = len(arr1)
        return list(map(list.__add__, arr1, arr2))

    def decode(self, arr):
        arr1 = []
        arr2 = []
        for item in arr:
            arr1.append(item[0:self.pivot])
            arr2.append(item[self.pivot:])
        return [arr1, arr2]

    def calc_residuals(self, arr1, arr2):
        distances = []
        assert len(arr1) == len(arr2)
        for i, pos in enumerate(arr1):
            distances.append(euclidean_distances([pos], [arr2[i]])[0][0])
        return distances


class PositionFiltering(object):
    """Assumes couples have been mirrored"""
    def __init__(self, max=.6):
        self.max = max

    def __repr__(self):
        return "PositionFiltering(max=" + str(self.max) + ")"

    def transform(self, couples):
        #we need to make a model and fetch our data
        #we hard code the most accurate model
        import data, preprocessing, wrappers, vector_math
        people = data.get.people_raw()
        people = preprocessing.people.Flatten().transform(people)
        Xy = data.make.couples_xy(couples, people)

        from sklearn.ensemble import GradientBoostingRegressor
        model = wrappers.SklearnWrapper(GradientBoostingRegressor(loss="quantile"))
        model.fit(*Xy)

        good_couples = []

        for couple in couples:
            source = couple["male"]
            target = couple["female"]

            soulmate = model.predict_for_single_point(people[source])
            rec_list = vector_math.get_rec(people, soulmate)
            index = [othername[0] for othername in rec_list].index(target)

            length = len(rec_list)
            max = length * self.max

            if index < max:
                good_couples.append(couple)

        return good_couples