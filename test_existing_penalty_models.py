"""see test_existing_models.py for explanation"""

import validator
import os, json
from sklearn.externals import joblib
from postprocessing import Average, MetricEqualizer


def gen_info(model):
    m = joblib.load(os.path.join("penalty_models", model))

    model_info = {}
    model_info["name"] = model

    print("")
    scores = []
    for _ in range(5):
        print("*", end="")
        scores.append(validator.penalty_val([m], [Average(), MetricEqualizer(metric="percentage")], swap)["score"])
    model_info[model] = scores

    print(model_info)
    with open(os.path.join("penalty_model_info", model + ".json"), "w") as f:
        f.write(json.dumps(model_info))

    return model_info

models = os.listdir("models")
models.sort()
info = list(map(gen_info, models))

with open("penalty_model_info.json", "w") as f:
    f.write(json.dumps(info))
