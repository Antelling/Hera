from sklearn.externals import joblib
import os
import colors

files = os.listdir("models")
files.sort()

for file in files:
    colors.blue("-----------------------------------------------------------------------")
    best = joblib.load(os.path.join("models", file))

    params = best.get_params()
    keys = list(params.keys())
    keys.sort()

    for key in keys:
        print(colors.underline(key) + ": " + str(params[key]).replace("\n", ""))

    print("\033[92mmodel\033[0m: " + str(params["regressor"].model))
    colors.red(file)

    input("")