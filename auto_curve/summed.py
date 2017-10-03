from scipy.optimize import curve_fit

from sklearn.base import BaseEstimator, RegressorMixin

from inspect import signature

import numpy as np


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def linear(x, a, b):
    return a * x + b


def logistic(x, max, steep, slide_hor, slide_ver):
    steep /= 3  # we want to slow down how fast scipy goes over this one, or it sometimes errors out
    return max / (1 + np.exp(-steep * (x + slide_hor))) + slide_ver


def quartic(x, a, b, c):
    return a * x ** 2 + b * x + c


def cubic(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 * c * x + d


def seminormal(x, center, left_spread, right_spread, slide_ver, stretch_ver):
    # normal with free-er variables
    left = 1 / np.sqrt(left_spread)
    right = ((x - center) ** 2) / (2 * right_spread)
    return slide_ver + stretch_ver * left * np.exp(right)


def logarithmic(x, slide_ver, stretch_ver, stretch_hor, slide_hor):
    return slide_ver + stretch_ver * np.log((x + slide_hor) * stretch_hor)


class SingleExtender(object):
    """Extends a math function to work over multidimensional X"""

    def __init__(self, function):
        self.function = function
        self.vars = len(signature(function).parameters) - 1

    def call(self, X, *opts):
        opts = list(opts)

        # split opts into function-opt sized chunks
        opts = list(chunks(opts, self.vars))

        return np.sum([self.function(X[i], *opt) for i, opt in enumerate(opts)])


class MultiExtender(object):
    """extends n math functions to work over n dimensions"""

    def __init__(self, functions):
        self.functions = functions
        self.vars = [len(signature(function).parameters) - 1 for function in functions]
        self.total_vars = np.sum(self.vars)

    def call(self, X, *opts):
        opts = list(opts)
        total = 0
        index = 0
        for i, f in enumerate(self.functions):
            total += f(X[i], *opts[index:index + self.vars[i]])
            index += self.vars[i]
        return total


def score(params, curve, X, y):
    # computes r_score
    residuals = y - curve(X, *params)
    residual_sum_of_squares = np.sum(residuals ** 2)
    total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    r_squared = float(r_squared)
    return r_squared


def safe_curve_fit(f, X, y, n_dimensions, maxfev=3000, method="lm"):
    fails = 0

    while True:
        try:
            p0 = np.random.uniform(-1, 1, n_dimensions)
            params, _ = curve_fit(f, X, y, p0=p0, maxfev=maxfev, method=method)
            break
        except (RuntimeError, ValueError):
            fails += 1
            if fails > 200:
                params = np.random.uniform(-1, 1, n_dimensions) #screw it, make up some numbers
                break
    return params


class SummedCurver(BaseEstimator, RegressorMixin):
    def __init__(self, maxfev=3000, method="lm"):
        self.maxfev = maxfev
        self.method = method

    def fit(self, X, y):
        rot = np.rot90(X)

        best_functions = []
        for row in rot:
            best_score = 0
            best_fun = None
            for fun in linear, logistic, quartic, cubic, seminormal, logarithmic:
                params = safe_curve_fit(fun, row, y, len(signature(fun).parameters) - 1, maxfev=self.maxfev,
                                        method=self.method)
                s = score(params, fun, row, y)
                if s > best_score:
                    best_score = s
                    best_fun = fun
            best_functions.append(best_fun)

        m = MultiExtender(best_functions)

        params = safe_curve_fit(m.call, rot, y, m.total_vars, maxfev=(self.maxfev * 5), method=self.method)

        self.m = m
        self.params = params

        return self

    def predict(self, X):
        return self.m.call(np.rot90(X), *self.params)
