import colors, data
from wrappers import SklearnWrapper
import numpy as np

# region setup
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics.pairwise import euclidean_distances


def dist_score(estimator, X, y):
    """Calculate residuals of estimator"""

    # okay so the pipeline object will do a lot of mirroring and filtering to couple objects
    # which we don't want to happen, we want to get the predictions
    # so we extract positions and pass those instead
    people = data.get.people_raw()
    positions = [people[couple["male"]]["position"] for couple in X]
    predictions = estimator.predict(positions)
    residuals = euclidean_distances(predictions, y)
    return -np.mean(residuals)


# endregion

# region models
from sklearn.pipeline import Pipeline
import preprocessing as pre
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from auto_curve import AutoCurver
# from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

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

    "aut": SklearnWrapper(MultiOutputRegressor(AutoCurver(maxfev=100000)), accept_singleton=True)
}

from sklearn.cluster import SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding
from sklearn.decomposition import PCA

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
    "sanitize": [
        pre.couples_xy.SanitizeStartEnd(),
        pre.transformers.Pass(),
    ],
    "sanitize__contamination": [.01, .03, .05, .07, .1, .15, .2, .3, .4, .5],
    "form_data__alg": [
        None,

        TSNE(n_components=3),
        TSNE(n_components=4),

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

param_grids = [
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
        'regressor__model__regressor__estimator__min_samples': [2],
    }),
    ("aut", {
        'regressor': [models["aut"]],
        "regressor__model__estimator__max_params": [5, 4, 3],
        "regressor__model__estimator__certainty_scaler": [.1, .25, .5, .75, 1, 1.2, 1.5, 1.75, 2, 2.5, 3, 4, 5]
    })
]

# endregion

# region pipeline

from sklearn.manifold import TSNE

pipeline = Pipeline([
    ("form_data", pre.transformers.FormData(TSNE(n_components=4))),
    ("standard", pre.people.Standard()),
    ("flatten", pre.people.Flatten()),
    ("erf", pre.people.Erf()),
    ("mirror", pre.couples_raw.Mirror()),  #TODO: check if females are being predicted correctly if this is None
    # ("adjust_time", pre.couples_raw.Time_mod()),  # opt
    # ("position_filter", pre.couples_raw.PositionFiltering()), #opt
    ("add_xy", pre.transformers.AddCouplesXy()),
    ("sanitize", pre.couples_xy.SanitizeStartEnd()),  # also start-end and vec, opt
    ("cluster", pre.couples_xy.Cluster()),  # using spectral embedding, try other methods, opt
    ("get_xy", pre.transformers.GetXy()),
    ("regressor", models["aut"])
])

# endregion

best_score = 10000
best_cv = {}

couples = data.get.couples_raw()
people = data.get.people_raw()
y = [people[couple["female"]]["position"] for couple in couples]

from sklearn.externals import joblib

scores = {}
for name, grid in param_grids:
    scores[name] = []

from validator import val
from postprocessing import MetricEqualizer, Average

while True:
    for name, param_grid in param_grids:
        colors.white("")
        param_grid = {**param_grid, **base_grid}
        try:
            rand_cv = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=7,
                scoring=dist_score,
                cv=7,
                return_train_score=False
            )
            rand_cv.fit(couples, y)

            score = val([rand_cv.best_estimator_], [Average(), MetricEqualizer(metric="percentage")])["score"]

            scores[name].append(score)

            if score < best_score:
                best_score = score
                best_cv = rand_cv
        except Exception as e:
            colors.red(e)

    joblib.dump(best_cv.best_estimator_, "models/" + str(round(best_score, 4)) + "-model.pkl")

    colors.white("_________________________")
    for name in scores:
        colors.blue(name + ": " + str(np.mean(scores[name])))
    colors.green(best_score)
    colors.white(best_cv.best_estimator_.get_params())
