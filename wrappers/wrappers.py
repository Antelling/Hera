"""Wrapper classes to provide inverse predicting, and ensure fitting doesn't remember old data."""

import numpy as np
from sklearn.base import clone
import vector_math, colors
from copy import deepcopy
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


class Wrapper(ABC):
    def predict(self, X):
        return self._fitted.predict(np.array(X))

    def predict_inverse(self, X):
        return self._fitted_inverse.predict(np.array(X))

    @abstractmethod
    def fit(self, X, y): pass


class SklearnWrapper(Wrapper, BaseEstimator):
    def __init__(self, model, accept_singleton=False, **params):
        self.model = model
        self.accept_singleton = accept_singleton
        self.set_params(**params)

    def fit(self, X, y):
        if self.accept_singleton:
            y = X[1]
            X = X[0]
        model = clone(self.model)
        inverse_model = clone(self.model)

        self._fitted = model.fit(X, y)
        self._fitted_inverse = inverse_model.fit(y, X)

    def predict(self, X):
        return self._fitted.predict(np.array(X))

    def predict_inverse(self, X):
        return self._fitted_inverse.predict(np.array(X))


class KerasWrapper(Wrapper):
    def __init__(self, model, epochs=1500, compile_params=None):
        self.model_definition = model
        self.epochs = epochs

        if compile_params is None:
            compile_params = {"loss": "mse", "optimizer": "adam"}
        self.compile_params = compile_params

        from keras.models import Sequential
        self.sequential = Sequential

    def fit(self, X, y):
        models = []
        for _ in range(0, 2):
            model = self.sequential(deepcopy(self.model_definition))
            model.compile(**self.compile_params)
            models.append(model)
        models[0].fit(X, y, epochs=self.epochs, verbose=False)
        models[1].fit(y, X, epochs=self.epochs, verbose=False)
        self.fitted = models[0]
        self.fitted_inverse = models[1]


class GenericWrapper(Wrapper):
    """Accepts two functions, the model creation function, and the prediction function. Model creation will receive X
    and y, and should return and object that will be passed to predict_func. Predict_func will receive the model and the
    person object it should predict, and should return a soulmate schema"""

    def __init__(self, model_func, predict_func):
        self.model_definition = model_func
        self.predict_func = predict_func

    def fit(self, X, y):
        self.fitted = self.model_definition(X, y)
        self.fitted_inverse = self.model_definition(y, X)

    def predict(self, people):
        return list(map(self.single_predict, people))

    def single_predict(self, person):
        return self.predict_func(self.fitted, person)

    def predict_inverse(self, people):
        return list(map(self.inverse_single_predict, people))

    def inverse_single_predict(self, person):
        return self.predict_func(self.fitted_inverse, person)
