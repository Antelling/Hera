import random
import numpy as np

import data, colors
from pipeline.penalty_pipeline import pipeline, param_grids
from postprocessing import MetricEqualizer, Average
from validator import penalty_val

from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib

from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances


# region gen data
def gen_penalty(months):
    if months > 23: return 1
    if months > 3: return .75
    return .5


couples = data.get.couples_raw()
people = data.get.people_raw()
people_xy, names = data.get.people_xy()

for i, couple in enumerate(couples):
    penalty = 0 if couple["married"] == True else gen_penalty(couple["length"])
    couple["penalty"] = penalty
    couple[i] = couple


def gen_training_data(couples):
    X, y = [], []

    for couple in couples:
        x = people[couple["male"]]["position"] + people[couple["female"]]["position"]
        X.append(x)
        y.append(couple["penalty"])

    return X, y


X, y = gen_training_data(couples)

class LightBulbRegressor(BaseEstimator):
    def __init__(self, degree=2):
        self.degree = 2
        self.sources = []

    def fit(self, X, y):
        for i, position in enumerate(X):
            self.sources.append([position, y[i]])

    def predict(self, X):
        return [self.predict_for_single_point(x) for x in X]

    def predict_for_single_point(self, position):
        total = 0
        for lightbulb in self.sources:
            total += lightbulb[1] / (self.get_distance(lightbulb[0], position)**2)
        return total

    def get_distance(self, a, b):
        return euclidean_distances([a,b])[0][1]

lbr = LightBulbRegressor()
lbr.fit(X, y)

print(lbr.predict([[1,1,1,1,1,1,1,1,1,1]]))


