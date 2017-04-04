"""We want to test, with and without RANSAC:
    * 3rd degree linear regression, Flattened, 15 spectral clusters, DistanceNormalized
    * quantile Gradient Boosting, Erf, Mirrored, 16 spectral clusters, ZScoreNormalized
"""

import colors

from wrappers import SklearnWrapper

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from preprocessing.people import Standard, Erf, Flatten

from preprocessing.couples_raw import Mirror, RANSAC

from preprocessing.couples_xy import Cluster
from sklearn.cluster import SpectralClustering

from postprocessing.normalize import MetricEqualizer
from postprocessing.average import Average

from validator import val

best_lin = val(
    people_pre=[Standard(), Erf()],
    couples_raw_pre=[Mirror()],
    couples_xy_pre=[Cluster(SpectralClustering(n_clusters=16))],
    alg_gen=[SklearnWrapper(GradientBoostingRegressor(loss="quantile"))],
    maps_post=[Average(), MetricEqualizer(metric="zscore")]
)

colors.green(best_lin["score"])

best_ransac_lin = val(
    people_pre=[Standard(), Erf()],
    couples_raw_pre=[Mirror(), RANSAC(min_consensus=40, max_iter=100)],
    couples_xy_pre=[],
    alg_gen=[SklearnWrapper(GradientBoostingRegressor(loss="quantile"))],
    maps_post=[Average(), MetricEqualizer(metric="zscore")]
)

colors.orange(best_lin["score"])
colors.green(best_ransac_lin["score"])