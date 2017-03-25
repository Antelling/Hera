from wrappers import GenericWrapper, SklearnWrapper
import random

def random_model(_,__):
    pass

def random_rec(model, person):
    return [random.sample(range(1, 100), 5), [1] * 5]

from sklearn.dummy import DummyRegressor

def gen_dummy():
    #yield GenericWrapper(random_model, random_rec)
    yield SklearnWrapper(DummyRegressor(strategy="mean"))