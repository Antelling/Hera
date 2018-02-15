#okay so penalty models randomly select bad examples for testing
#but they can't select ALL the possible bad pairings, since A, that would take literally forever, and B, that would
#dwarf the good examples
#but the random selection and relatively small sample size can lead to drastically different results every time it is
#run. So to mitigate that, here we load n copies of the top n models, train each on different randomly selected bad
#couples, and average the results of all of those to form recs

import data, copy, os, json, random
from validator import score
from sklearn.externals import joblib
import numpy as np
from sklearn.base import BaseEstimator
from scipy.stats import hmean, tmean
from postprocessing import MetricEqualizer, CoupleEqualizerFast


n = 5

#region gen data
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
    for _ in range(len(couples) * 7):
        m = random.choice(people_xy)
        f = random.choice(people_xy)
        vector = m + f
        if vector not in X:
            X.append(vector)
            y.append(6)
    return X, y

#endregion

#region ghetto ensemble

#we can't use the default sklearn ensembles since they all want to use the same data
#plus they confuse me

class GhettoEnsemble(BaseEstimator):
    def __init__(self, models, mean):
        self.models = models

        #method is a function that produces the average
        #by letting it be user definable we allow for harmonic means and that sort of thing
        self.mean = mean

    def predict(self, X):
        predicted_x = []
        for model in self.models:
            predicted_x.append(model.predict(X))
        predicted_x = self.mean(predicted_x, axis=1)
        return predicted_x

#endregion

#first we load the top n penalty models


info = []
for file in os.listdir("penalty_model_info"):
    d = json.loads(open(os.path.join("penalty_model_info", file)).read())
    name = d["name"]
    d = np.mean(d[name])
    info.append([name, d])

#then sort them, take the top n, train them on random data, and add it to our list
top_n = []
info.sort(key=lambda x:x[1])
best_model = joblib.load(os.path.join("penalty_models", info[0][0]))
for index in range(n):
    top_n.append(copy.deepcopy(best_model))

model = GhettoEnsemble(top_n, tmean)

post = [
            MetricEqualizer(metric="percentage"),
            CoupleEqualizerFast(),
            MetricEqualizer(metric="zscore")
        ]

best = {
    "alg": {},
    "score": 9999,
    "post": ""
}

total_score_map = {}
alg = {}
for i, couple in enumerate(couples):
    print(couple["male"] + "/" + couple["female"])

    new_couples = copy.deepcopy(couples)
    top_n = copy.deepcopy(top_n)
    del new_couples[i]

    for model in top_n:
        model.fit(*gen_training_data(new_couples))

    alg = GhettoEnsemble(top_n, tmean)

    maps = {
        "scoreable": {"average": {}},  # we can't have one-way maps with this strategy
        "misc": {}  # eg mono-couples, pop-list
    }

    for name in people:
        person = people[name]
        penalties = []
        print(name + "...", end="", flush=True)
        for othername in people:
            otherperson = people[othername]
            vec = person["position"] + otherperson["position"]
            pen = alg.predict([vec])
            penalties.append([othername, pen])
        penalties.sort(key=lambda x: x[1])
        maps["scoreable"]["average"][name] = penalties

    for processor in post:
        maps = processor.transform(maps)

    score_map = score.score(maps, couple)
    for m in score_map:
        if not m in total_score_map:
            total_score_map[m] = []
        total_score_map[m] += score_map[m]

for m in total_score_map:
    total_score_map[m] = [np.percentile(total_score_map[m], 80),
                          np.mean(total_score_map[m]),
                          np.median(total_score_map[m])]

for m in total_score_map:
    if total_score_map[m][1] < best["score"]:
        best["score"] = total_score_map[m][1]
        best["alg"] = alg
        best["post"] = m

print(best)
