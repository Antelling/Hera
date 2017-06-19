from sklearn.base import BaseEstimator
from scipy.optimize import curve_fit
import numpy as np
import math, inspect


# region helpers
def sign(x):
    return -1 if x < 0 else 1


# endregion


# region poly

def linear(x, a, b):
    return a * x + b


def exponential(x, a, b, c):
    return a * x ** 2 + b * x + c


def cubic(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 * c * x + d


def quartic(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


def a_to_x(x, a, b, c):
    return a ** (x + b) + c


def sqrt(X, a, b, c):
    return [sign(x) * a * np.sqrt(np.abs(x)) + b * np.abs(x) + c for x in X]


def cbrt(X, a, b, c, d):
    return [sign(x) * a * np.cbrt(np.abs(x)) + b * np.sqrt(np.abs(x)) + c * np.abs(x) + d for x in X]


# endregion
# region distributions
def normal(x, center, spread, slide_ver, stretch_ver):
    left = 1 / np.sqrt(2 * math.pi * spread)
    right = - ((x - center) ** 2) / (2 * spread)
    return slide_ver + stretch_ver * (left * np.exp(right))


def neg_normal(x, center, spread, slide_ver, stretch_ver):
    # so like mathematically these are the same because stretch_ver can be negative
    # but in reality scipy's curve_fit sometimes chokes when flipping the normal, so we have this upside down one
    left = 1 / np.sqrt(2 * math.pi * spread)
    right = ((x - center) ** 2) / (2 * spread)
    return slide_ver + stretch_ver * left * np.exp(right)


def laplace(x, u, b, ver_slide, ver_stretch):
    y = 1 / (2 * b) * np.exp(-np.abs(x - u) / b)
    return y * ver_stretch + ver_slide


def cauchy(x, u, l, ver_slide, ver_stretch):
    y = (1 / (math.pi * l)) * (l ** 2 / ((x - u) ** 2 + l ** 2))
    return y * ver_stretch + ver_slide


# endregion
# region periodic
def sinusoidal(x, slide_ver, slide_hor, stretch_hor, stretch_ver):
    internal = x * stretch_hor + slide_hor
    return slide_ver + stretch_ver * np.sin(internal)


def sin_plus_normal(x, a, b, c, d, u, s, e, f):
    return sinusoidal(x, a, b, c, d) + neg_normal(x, u, s, e, f)


def sin_times_x(x, a, b, c, d, u, s):
    return sinusoidal(x, a, b, c, d) * linear(x, u, s)


def sin_plus_x(x, a, b, c, d, u, s):
    return sinusoidal(x, a, b, c, d) + linear(x, u, s)


# endregion
# region other
def logistic(x, max, steep, slide_hor, slide_ver):
    steep /= 3  # we want to slow down how fast scipy goes over this one, or it sometimes errors out
    return max / (1 + np.exp(-steep * (x + slide_hor))) + slide_ver


def logistic2(x, max, steep, slide_hor, slide_ver):
    steep += 1  # even with slowing, it sometimes flies to infinity, so we have this one with slightly different params
    return -max / (1 + np.exp(-steep * (x + slide_hor))) + slide_ver


function_lists = {
    "all": [linear, exponential, cubic, quartic, a_to_x, sqrt, cbrt, normal, neg_normal, laplace, cauchy, sinusoidal,
            sin_plus_normal, sin_times_x, sin_plus_x, logistic, logistic2],
    "poly": [linear, exponential, cubic],
    "common": [linear, exponential, logistic, logistic2, normal]
}
# endregion



class AutoCurver(BaseEstimator):
    def __init__(self, max_params=8, maxfev=100000, certainty_scaler=1, function_type="common"):
        self.max_params = max_params
        self.function_type = function_type
        functions = function_lists[function_type]

        self.estimators = []
        self.functions = []
        for f in functions:
            if len(inspect.signature(f).parameters) <= max_params:
                self.functions.append(f)

        self.maxfev = maxfev
        self.certainty_scaler = certainty_scaler
        pass

    def fit(self, X, y):
        # produce a new estimator for each dimension
        # an estimator is a dictionary of the function, its params, and its rscore
        dimensions = X.shape[1]
        for dimension in range(dimensions):
            x = [x[dimension] for x in X]
            self.estimators.append(self._fit_on_one_dimension(x, y))
        return self

    def predict(self, X):
        # for every point, get a prediction and r2, then compute the weighted
        Y = []
        for point in X:
            new_point = []
            for i, dimension in enumerate(point):
                e = self.estimators[i]
                y = e["f"](np.array([dimension]), *e["params"])[0]
                new_point.append((y, e["score"] ** self.certainty_scaler))
            Y.append(self._weighted_average(new_point))
        return Y

    def _weighted_average(self, point):
        total = np.sum([x[1] for x in point])
        weights = [x[1] / total for x in point]
        data = [x[0] for x in point]
        return np.average(data, weights=weights)

    def _fit_on_one_dimension(self, x, y):
        np.seterr(all="ignore")
        # find the best function for the x, y data
        best = {"params": [], "score": 0, "f": {}}
        for f in self.functions:
            try:
                score, params = self._fit_func(f, x, y)
                if score > best["score"]:
                    best = {"params": params, "score": score, "f": f}
            except RuntimeError:
                # optimal params not found - maxfev needs increased
                print(f.__name__ + " not fit, increase maxfev!")
        return best

    def _fit_func(self, f, x, y):
        x = np.array(x)
        y = np.array(y)
        # fit the function, then return rsquared and params
        fitted_params, _ = curve_fit(f, x, y, maxfev=self.maxfev)

        residuals = y - f(x, *fitted_params)
        residual_sum_of_squares = np.sum(residuals ** 2)
        total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
        r_squared = float(r_squared)
        return r_squared, fitted_params
