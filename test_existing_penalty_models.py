"""see test_existing_models.py for explanation"""

import validator, data, random
import os, json
from sklearn.externals import joblib
from postprocessing import CoupleEqualizerFast, MetricEqualizer


def gen_penalty(months):
    if months > 23: return 1
    if months > 3: return 2
    return 3


couples = data.get.couples_raw()
people = data.get.people_raw()
people_xy, names = data.get.people_xy()

for i, couple in enumerate(couples):
    penalty = 0 if couple["married"] == True else gen_penalty(couple["length"])
    couple["penalty"] = penalty
    couple[i] = couple


def gen_training_data(couples):
    X, y = [], []

    for couple in couples:
        x = people[couple["male"]]["position"] + people[couple["female"]]["position"]
        X.append(x)
        y.append(couple["penalty"])

    # now we need some negative examples
    for _ in range(len(couples) * 3):
        m = random.choice(people_xy)
        f = random.choice(people_xy)
        vector = m + f
        if vector not in X:
            X.append(vector)
            y.append(6)
    return X, y




def gen_info(model):
    m = joblib.load(os.path.join("penalty_models", model))

    model_info = {}
    model_info["name"] = model

    print("")
    tests = []
    for _ in range(5):
        X, y = gen_training_data(couples)
        print("*", end="")
        tests.append(validator.penalty_val(m, X, y, couples, [
            MetricEqualizer(metric="percentage"),
            CoupleEqualizerFast(),
            MetricEqualizer(metric="zscore")
        ]))
    model_info[model] = tests

    print(model_info)
    with open(os.path.join("detailed_penalty_model_info", model + ".json"), "w") as f:
        f.write(json.dumps(model_info))

    return model_info

models = os.listdir("penalty_models")
models.sort()
info = list(map(gen_info, models))

with open("penalty_model_info.json", "w") as f:
    f.write(json.dumps(info))
