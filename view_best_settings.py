from sklearn.externals import joblib
import os
import colors
import json
import numpy as np

info = json.loads(open("model_info.json").read())

m2f = []
f2m = []

for model in info:
    m2f.append((model["name"], np.mean(model["m2f"])))
    f2m.append((model["name"], np.mean(model["f2m"])))

m2f.sort(key=lambda x:x[1])
f2m.sort(key=lambda x:x[1])

for i, m in enumerate(m2f):
    colors.blue("-----------------------------------------------------------------------")
    best = joblib.load(os.path.join("models", m[0]))

    params = best.get_params()
    keys = list(params.keys())
    keys.sort()

    for key in keys:
        print(colors.underline(key) + ": " + str(params[key]).replace("\n", ""))

    print("\033[92mmodel\033[0m: " + str(params["regressor"].model))
    colors.red("m2f: " + str(m[1]))
    colors.red("f2m: " + str(f2m[i][1]))

    input("")