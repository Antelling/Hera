from wrappers import SklearnWrapper
from sklearn.pipeline import Pipeline
import preprocessing as pre

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from auto_curve import SummedCurver, WeightedCurver
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.cluster import SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding
from sklearn.decomposition import PCA

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator

#initiated sklearn regressor objects and a couple polynomial pipelines
models = {
    "svr": SVR(),
    "kr": KernelRidge(),
    "rf": RandomForestRegressor(),
    "gb": GradientBoostingRegressor(),

    "lr": Pipeline([
        ("poly", PolynomialFeatures(2)),
        ("regressor", LinearRegression()),
        ]),
    "hr": Pipeline([
        ("poly", PolynomialFeatures(2)),
        ("regressor", HuberRegressor())
    ]),
    "ran": Pipeline([
        ("poly", PolynomialFeatures(2)),
        ("regressor", RANSACRegressor())
    ]),

    "gpr": GaussianProcessRegressor(),

    "wei": WeightedCurver(maxfev=100000),
    "sum": SummedCurver(maxfev=2000, method="dogbox"),
}

#params that every model will share, like dimensionality reduction options
base_grid = {
    "standard": [StandardScaler(), None],
    #"cluster": [
    #    pre.couples_xy.Cluster(SpectralClustering()),
    #    pre.couples_xy.Cluster(MiniBatchKMeans()),
    #    pre.couples_xy.Cluster(AgglomerativeClustering()),
    #    pre.couples_xy.Cluster(Birch()),
    #    pre.transformers.Pass(),
    #],
    #"cluster__clusterer__n_clusters": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26],
    #"cluster__replace": [True],
    #"sanitize": [
    #    pre.couples_xy.SanitizeStartEnd(),
    #    pre.transformers.Pass(),
    #],
    #"sanitize": [IsolationForest(), EllipticEnvelope()],
    #"sanitize__contamination": [.01, .03, .05, .07, .1, .15, .2, .3, .4, .5],
    "decomp": [
        None,  # I have to explicitly specify n_components for each because of this
        PCA(),
    ],
    #"flatten": [pre.people.Flatten(), None],
}

#model specific params
model_grids = [
    ("svr", {
        'regressor': [models["svr"]],
        'regressor__C': [.01, .1, .5, .75, 1, 1.5, 2, 5, 10],
        'regressor__epsilon': [.01, .1, .5, .75, 1, 1.5, 2, 5, 10],
        'regressor__kernel': ["linear", "rbf", "sigmoid"],
    }),
    ("kr", {
        'regressor': [models["kr"]],
        'regressor__kernel': ["linear", "rbf", "laplacian", "sigmoid"],
        'regressor__gamma': [.01, .05, .02, .1, .2, .3, .5, .75, None],
    }),
    ("rf", {
        'regressor': [models['rf']],
        'regressor__n_estimators': [10, 20, 40],
        'regressor__criterion': ["mae", "mse"],
        'regressor__max_features': ["auto", "sqrt", None],
    }),
    ("gb", {
        'regressor': [models['gb']],
        'regressor__loss': ["ls", "lad", "huber", "quantile"],
        'regressor__max_depth': [2, 3, 4, 5],
        'regressor__max_features': ["auto", "sqrt", None],
    }),
    ("lin", {
        'regressor': [models['lr']],
        'regressor__poly__degree': [1, 2, 3, 4],

    }),
    ("hub", {
        'regressor': [models["hr"]],
        'regressor__poly__degree': [1, 2, 3, 4],
        'regressor__regressor__epsilon': [1, 1.1, 1.2, 1.3, 1.35, 1.4],
        'regressor__regressor__alpha': [.0001, .0002, .00005, .001, .01],
    }),
    ("ran", {
        'regressor': [models["ran"]],
        'regressor__poly__degree': [1, 2, 3, 4],
        'regressor__regressor__min_samples': [2, 3, 4],
    }),
    ("gpr", {
        "regressor": [models["gpr"]],
        "regressor__alpha": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]
    })
    # ("wei", {
    #    'regressor': [models["wei"]],
    #    "regressor__model__estimator__max_params": [5, 4, 3],
    #    "regressor__model__estimator__certainty_scaler": [.1, .25, .5, .75, 1, 1.2, 1.5, 1.75, 2, 2.5, 3, 4, 5]
    # }),
    # ("sum", {
    #    'regressor': [models["sum"]]
    # })
]

#now we combine the options
param_grids = []
for name, mg in model_grids:
    param_grids.append((name, {**base_grid, **mg}))

#the actual pipeline object
pipeline = Pipeline([
    ("decomp", PCA()),
    ("standard", pre.people.Standard()),
    #("flatten", pre.people.Flatten()),
    #("sanitize", IsolationForest()),
    #("cluster", pre.couples_xy.Cluster()),
    ("regressor", models["sum"])
])