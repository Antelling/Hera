"""Okay so under models we have all of these models that were found to be good by find_pre.
However, the scores of these models is based off of one trial. There's a lot of randomness.
Additionally, if we add new couples, we want to be able to evaluate the new scores.

So in this file, we are going to load every model, then run it numerous times with a LOU validator on the existing
 couples."""

import validator
import os, json
from sklearn.externals import joblib
from postprocessing import Average, MetricEqualizer


def gen_info(model):
    m = joblib.load(os.path.join("models", model))

    model_info = {}
    model_info["name"] = model

    for swap in True, False:
        print("")
        scores = []
        for _ in range(5):
            print("*", end="")
            scores.append(validator.val([m], [Average(), MetricEqualizer(metric="percentage")], swap)["score"])
        key = "f2m" if swap else "m2f"
        model_info[key] = scores

    print(model_info)
    with open(os.path.join("model_info", model + ".json"), "w") as f:
        f.write(json.dumps(model_info))

    return model_info

models = os.listdir("models")
models.sort()
info = list(map(gen_info, models))

with open("model_info.json", "w") as f:
    f.write(json.dumps(info))
