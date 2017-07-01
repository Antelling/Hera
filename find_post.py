# we test every normalization strategy

from sklearn.externals import joblib

import colors

model = joblib.load("model.pkl")

from postprocessing import CoupleEqualizerFast, MetricEqualizer, Average

from validator import val

while True:
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
