import numpy as np
from wrappers import GenericWrapper

def empty_model(_,__):
    pass

def sim_rec(model, person):
    return [person["position"], [1] * len(person["position"])]

def mirror_about_mean(model, person):
    return [(np.array(person["position"]) * -1).tolist(), [1] * len(person["position"])]

def gen_naive():
    yield GenericWrapper(empty_model, sim_rec)
    yield GenericWrapper(empty_model, mirror_about_mean)