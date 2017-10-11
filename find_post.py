# we test every normalization strategy

from sklearn.externals import joblib
from sklearn.base import BaseEstimator, RegressorMixin

import os

import colors

model = joblib.load("model.pkl")

from postprocessing import CoupleEqualizerFast, MetricEqualizer, Average

from validator import val

class SimModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        pass

    def predict(self, X):
        return X

print("testing sim score")
local = val(
        algs=[SimModel()],
        post=[
            Average()
        ]
    )

colors.purple(local)
colors.green(local["score"])

for model in os.listdir("models"):
    model = joblib.load(os.path.join("models", model))
    local = val(
        algs=[model],
        post=[
            Average(),
            CoupleEqualizerFast(),
            MetricEqualizer(metric="zscore"),
            MetricEqualizer(metric="percentage"),
            MetricEqualizer(metric="zscore_median"),
            MetricEqualizer(metric="percentage_median"),
            MetricEqualizer(metric="distance"),
            MetricEqualizer(metric="distance_median")
        ]
    )

    colors.purple(local)
    colors.green(local["score"])
