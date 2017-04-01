"""Holds wrapper classes for algorithm implementations. Each wrapper class should do the following:
1. __init__ accepts the model to be used (eg, a sklearn regressor, a keras neural net, or something else), and saves
    it internally
2. fit receives data in the couples_xy format, and internally saves a predictive model trained from this data
3. the methods predict and predict_for_single_point return the predicted soulmates for a list of points or single point,
respectively

.fit should always delete the previous trained model, and then train a new one on only the provided data

All lists, both input and output, are normal python arrays, NOT numpy arrays.

Note that the "soulmate" format is not just a 5 dimensional position, it is a two item tuple of form:
    (position, dimension importance). Dimension importance is later used to scale all the data, before finding the
    closest matches.
        Note that ideally, relative dimension importance wouldn't be a linear transformation, but something like a large
        polynomial. But I have no possible method of finding that ideal form, I can barely approximate linear importance
"""

import numpy as np
from sklearn.base import clone
import vector_math, colors


class SklearnWrapper(object):
    def __init__(self, model, params=None, scale_importance=False):
        self.scale_importance = scale_importance
        self.model = model

    def __repr__(self):
        return str(self.model) + " scale_importance:" + str(self.scale_importance)

    def create_models(self, data):
        models = []
        for i in range(0, 5):
            model_copy = clone(self.model)  # we clone to prevent fitting-in-place from remembering past data
            models.append(model_copy.fit(np.array(data["X"], dtype="float_"), np.array(data["y"][i], dtype="float_")))
        return models

    def fit(self, X, y):
        # okay we need to produce a male and female model
        # X and y are both lists of points

        male = {"X": [], "y": [[], [], [], [], []]}
        female = {"X": [], "y": [[], [], [], [], []]}
        for i, x in enumerate(X):
            male["X"].append(X[i])
            female["X"].append(y[i])
            for j in range(0, 5):
                try:
                    male["y"][j].append(y[i][j])
                except IndexError:
                    colors.red(y)
                    colors.orange(y[i])
                    colors.blue(j)
                female["y"][j].append(X[i][j])
        male_models = self.create_models(male)
        female_models = self.create_models(female)
        self.trained_models = [male_models, female_models]

    def predict(self, people):
        return list(map(self.predict_for_single_point, people))

    def predict_for_single_point(self, person):
        soulmate = []
        model_index = 0 if person["gender"] is "male" else 1
        model_to_use = self.trained_models[model_index]
        for dimension in range(0, 5):
            soulmate.append(model_to_use[dimension].predict([person["position"]]).tolist()[0])
        if self.scale_importance:
            importance = vector_math.make_relative_importance(person["position"], soulmate)
        else:
            importance = [1] * 5
        return [soulmate, importance]


class KerasWrapper(object):
    def __init__(self, model, epochs=1500, compile_params=None, scale_importance=False):
        self.model_definition = model
        self.scale_importance = scale_importance
        self.epochs = epochs

        if compile_params is None:
            compile_params = {"loss": "mse", "optimizer": "adam"}
        self.compile_params = compile_params

        from keras.models import Sequential
        self.sequential = Sequential

    def fit(self, X, y):
        models = []
        for _ in range(0, 2):
            model = self.sequential(self.model_definition)
            model.compile(**self.compile_params)
            models.append(model)
        models[0].fit(X, y, nb_epoch=self.epochs, verbose=False)
        models[1].fit(y, X, nb_epoch=self.epochs, verbose=False)
        self.trained_models = models

    def predict(self, people):
        return list(map(self.predict_for_single_point, people))

    def predict_for_single_point(self, person):
        model_to_use = self.trained_models[0 if person["gender"] is "male" else 1]
        position = person["position"]
        position = np.array([position])
        soulmate = model_to_use.predict(position).tolist()[0]
        position = position.tolist()[0]
        if self.scale_importance:
            importance = vector_math.make_relative_importance(position, soulmate)
        else:
            importance = [1] * 5
        return [soulmate, importance]

    def __repr__(self):
        return str(self.model_definition) + " scale_importance" + str(self.scale_importance)


class GenericWrapper(object):
    """Accepts two functions, the model creation function, and the prediction function. Model creation will receive X
    and y, and should return and object that will be passed to predict_func. Predict_func will receive the model and the
    person object it should predict, and should return a soulmate schema"""

    def __init__(self, model_func, predict_func):
        self.model_func = model_func
        self.predict_func = predict_func

    def fit(self, X, y):
        self.model = self.model_func(X, y)

    def predict(self, people):
        return list(map(self.predict_for_single_point, people))

    def predict_for_single_point(self, person):
        return self.predict_func(self.model, person)
