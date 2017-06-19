from sklearn.preprocessing import StandardScaler
import data as d
import numpy as np
import random, colors
from abc import ABC
from sklearn.base import BaseEstimator


class PeopleBase(ABC, BaseEstimator):
    def transform(self, data):
        people = data["people"]
        people_xy = d.make.people_xy(people)
        people_xy[0] = self.transform_x(people_xy[0])
        for i, name in enumerate(people_xy[1]):
            people[name]["position"] = people_xy[0][i]
        data["people"] = people
        return data

    def transform_x(self, X):
        return X

    def fit_x(self, X):
        return X

    def fit(self, data, *_):
        people = data["people"]
        people_xy = d.make.people_xy(people)
        self.fit_x(people_xy[0])
        return self


class Standard(PeopleBase):
    def transform_x(self, X):
        transformed = self.scaler.transform(X)
        return transformed

    def fit_x(self, X):
        scaler = StandardScaler()
        self.scaler = scaler.fit(X)


class Erf(PeopleBase):
    def transform_x(self, X):
        import scipy.special as special
        return special.erf(X).tolist()


class Flatten(PeopleBase):
    def transform_x(self, X):
        X = np.array(X)
        X = np.rot90(X, k=1)
        X = list(map(self.flatten_dimension, X))
        X = np.rot90(X, k=3)
        return X.tolist()

    def flatten_dimension(self, d):
        n = len(d)
        step = 100 / n
        numbers = np.arange(0, 100, step)

        # okay so we want to snap our d to numbers
        # but we need to remember our original position
        # and we need to fuzz the numbers to avoid bad correlations
        structured_d = []
        for i, value in enumerate(d):
            structured_d.append({"index": i, "value": value + random.uniform(-.4, .4)})

        structured_d.sort(key=lambda x: x["value"])

        for i, value in enumerate(structured_d):
            structured_d[i]["flattened_value"] = numbers[i]

        structured_d.sort(key=lambda x: x["index"])

        d = []
        for value in structured_d:
            d.append(value["flattened_value"])
        return d


class Decompose(PeopleBase):
    def __init__(self, decomp=None, **params):
        if decomp is None:
            from sklearn.decomposition import PCA
            decomp = PCA(4)
        self.decomp = decomp
        self.set_params(**params)

    def __repr__(self):
        return "Decomp(" + str(self.decomp) + ")"

    def fit_x(self, X):
        self.decomp.fit(X)

    def transform_x(self, X):
        return self.decomp.transform(X)
