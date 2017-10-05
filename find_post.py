# we test every normalization strategy

from sklearn.externals import joblib

import os

import colors

model = joblib.load("model.pkl")

from postprocessing import CoupleEqualizerFast, MetricEqualizer, Average

from validator import val

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
