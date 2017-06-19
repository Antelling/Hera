import copy, wrappers
import numpy as np

X = [
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10],
]

y = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

from sklearn.svm import SVR

params = {
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__epsilon": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__kernel": ["rbf", "linear", "sigmoid"]
}
from sklearn.model_selection import RandomizedSearchCV


def scoring(estimator, X, y):
    print(y)
    resids = estimator.predict(X)
    print(resids)
    resids = np.sum([x**2 for x in resids - y])
    print(resids)
    return resids

cv = RandomizedSearchCV(wrappers.SklearnWrapper(SVR()), param_distributions=params, n_iter=40, scoring=scoring)
cv.fit(X, y)
