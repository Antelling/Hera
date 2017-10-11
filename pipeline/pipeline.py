from wrappers import SklearnWrapper
from sklearn.pipeline import Pipeline
import preprocessing as pre
from sklearn.multioutput import MultiOutputRegressor

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

#initiated sklearn regressor objects and a couple polynomial pipelines
models = {
    "svr": SklearnWrapper(MultiOutputRegressor(SVR()), accept_singleton=True),
    "kr": SklearnWrapper(KernelRidge(), accept_singleton=True),
    "rf": SklearnWrapper(RandomForestRegressor(), accept_singleton=True),
    "gb": SklearnWrapper(MultiOutputRegressor(GradientBoostingRegressor()), accept_singleton=True),

    "lr": SklearnWrapper(Pipeline([
        ("poly", PolynomialFeatures(2)),
        ("regressor", MultiOutputRegressor(LinearRegression()))
    ]), accept_singleton=True),
    "hr": SklearnWrapper(Pipeline([
        ("poly", PolynomialFeatures(2)),
        ("regressor", MultiOutputRegressor(HuberRegressor()))
    ]), accept_singleton=True),
    "ran": SklearnWrapper(Pipeline([
        ("poly", PolynomialFeatures(2)),
        ("regressor", MultiOutputRegressor(RANSACRegressor()))
    ]), accept_singleton=True),

    "gpr": SklearnWrapper(MultiOutputRegressor(GaussianProcessRegressor()), accept_singleton=True),

    "wei": SklearnWrapper(MultiOutputRegressor(WeightedCurver(maxfev=100000)), accept_singleton=True),
    "sum": SklearnWrapper(MultiOutputRegressor(SummedCurver(maxfev=2000, method="dogbox")), accept_singleton=True),
}

#params that every model will share, like dimensionality reduction options
base_grid = {
    "standard": [pre.people.Standard(), None],
    "mirror": [pre.couples_raw.Mirror(), None],
    "cluster": [
        pre.couples_xy.Cluster(SpectralClustering()),
        pre.couples_xy.Cluster(MiniBatchKMeans()),
        pre.couples_xy.Cluster(AgglomerativeClustering()),
        pre.couples_xy.Cluster(Birch()),
        pre.transformers.Pass(),
    ],
    "cluster__clusterer__n_clusters": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26],
    "cluster__replace": [True],
    "sanitize": [
        pre.couples_xy.SanitizeStartEnd(),
        pre.transformers.Pass(),
    ],
    "sanitize__alg": [IsolationForest(), EllipticEnvelope()],
    "sanitize__contamination": [.01, .03, .05, .07, .1, .15, .2, .3, .4, .5],
    "form_data__alg": [
        None,  # I have to explicitly specify n_components for each because of this

        TSNE(n_components=3),

        LocallyLinearEmbedding(n_components=3),
        LocallyLinearEmbedding(n_components=4),

        Isomap(n_components=3),
        Isomap(n_components=4),

        MDS(n_components=3),
        MDS(n_components=4),

        SpectralEmbedding(n_components=3),
        SpectralEmbedding(n_components=4),

        PCA(n_components=3),
        PCA(n_components=4),
    ],
    "flatten": [pre.people.Flatten(), None],
    "erf": [pre.people.Erf(), None]
}

#model specific params
model_grids = [
    ("svr", {
        'regressor': [models["svr"]],
        'regressor__model__estimator__C': [.01, .1, .5, .75, 1, 1.5, 2, 5, 10],
        'regressor__model__estimator__epsilon': [.01, .1, .5, .75, 1, 1.5, 2, 5, 10],
        'regressor__model__estimator__kernel': ["linear", "rbf", "sigmoid"],
    }),
    ("kr", {
        'regressor': [models["kr"]],
        'regressor__model__kernel': ["linear", "rbf", "laplacian", "sigmoid"],
        'regressor__model__gamma': [.01, .05, .02, .1, .2, .3, .5, .75, None],
    }),
    ("rf", {
        'regressor': [models['rf']],
        'regressor__model__n_estimators': [10, 20, 40],
        'regressor__model__criterion': ["mae", "mse"],
        'regressor__model__max_features': ["auto", "sqrt", None],
    }),
    ("gb", {
        'regressor': [models['gb']],
        'regressor__model__estimator__loss': ["ls", "lad", "huber", "quantile"],
        'regressor__model__estimator__max_depth': [2, 3, 4, 5],
        'regressor__model__estimator__max_features': ["auto", "sqrt", None],
    }),
    ("lin", {
        'regressor': [models['lr']],
        'regressor__model__poly__degree': [1, 2, 3, 4],

    }),
    ("hub", {
        'regressor': [models["hr"]],
        'regressor__model__poly__degree': [1, 2, 3, 4],
        'regressor__model__regressor__estimator__epsilon': [1, 1.1, 1.2, 1.3, 1.35, 1.4],
        'regressor__model__regressor__estimator__alpha': [.0001, .0002, .00005, .001, .01],
    }),
    ("ran", {
        'regressor': [models["ran"]],
        'regressor__model__poly__degree': [1, 2, 3, 4],
        'regressor__model__regressor__estimator__min_samples': [2, 3, 4],
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
    ("form_data", pre.transformers.FormData(TSNE(n_components=3))),
    ("standard", pre.people.Standard()),
    ("flatten", pre.people.Flatten()),
    ("erf", pre.people.Erf()),
    ("mirror", pre.couples_raw.Mirror()),
    ("add_xy", pre.transformers.AddCouplesXy()),
    ("sanitize", pre.couples_xy.SanitizeStartEnd()),  # also start-end and vec, opt
    ("cluster", pre.couples_xy.Cluster()),
    ("get_xy", pre.transformers.GetXy()),
    ("regressor", models["sum"])
])