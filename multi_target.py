"""okay so the pipeline and find_pre files take a position and output another position:
    [a, b, c, d, e] -> [z, y, x, w, v]
there are two issues with that:
    sometimes one target personality type isn't enough
    the algorithms like to play it safe and tell everyone to date the popular kids
So, instead of one target position, we are going to have a penalty
    [a, b, c, d, e] -> p
The more compatible you are with someone, the lower the penalty will be.
For our training data, the penalty will be determined by:
    married couple: 0
    dating for more than two years or I like you: 1
    dating for more than 3 months: 2
    dating: 3
    every other pair: 6
we will use a pipeline that includes all of the data preprocessing but doesn't have the regressor tail"""

import random
import numpy as np

import data, colors
from pipeline.penalty_pipeline import pipeline, param_grids
from postprocessing import MetricEqualizer, Average
from validator import penalty_val

from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib


# region gen data
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


X, y = gen_training_data(couples)

# endregion

while True:

    best_score = 10000
    best_cv = {}

    scores = {}
    for name, grid in param_grids:
        scores[name] = []

    for _ in range(7):
        for name, param_grid in param_grids:
            try:
                rand_cv = RandomizedSearchCV(
                    pipeline,
                    param_distributions=param_grid,
                    n_iter=7,
                    cv=7,
                    return_train_score=False
                )
                rand_cv.fit(X, y)

                print(rand_cv.best_score_)
                score = penalty_val(rand_cv.best_estimator_, X, y, couples, [MetricEqualizer(metric="percentage")])[
                    "score"]
                print(score)

                scores[name].append(score)

                if score < best_score:
                    best_score = score
                    best_cv = rand_cv
            except Exception as e:
                colors.red(e)

        joblib.dump(best_cv.best_estimator_, "penalty_models/" + str(round(best_score, 2)) + ".pkl")

        colors.white("_________________________")
        for name in scores:
            colors.blue(name + ": " + str(np.mean(scores[name])))
        colors.green(best_score)
        colors.white(best_cv.best_estimator_.get_params())



"""
pairings = {}
for name in people:
    person = people[name]
    penalties = []
    for othername in people:
        otherperson = people[othername]
        vec = person["position"] + otherperson["position"]
        pen = rf.predict([vec])[0].tolist()
        penalties.append([othername, pen])
    penalties.sort(key=lambda x: x[1])
    pairings[name] = penalties

import postprocessing
maps = {}
maps["scoreable"] = {}
maps["scoreable"]["average"] = pairings
pairings = postprocessing.MetricEqualizer(metric="percentage", name="main").transform(maps)["scoreable"]["main"]

import json
people_in_relationships = []
for couple in data.get.couples_raw():
    if couple["still_dating"]:
        people_in_relationships.append(couple["male"])
        people_in_relationships.append(couple["female"])
map_to_save = {
    "map": pairings,
    "people_raw": people,
    "list": names,
    "people_in_relationships": people_in_relationships
}

open("display_server/static/map.json", 'w').write(json.dumps(map_to_save))
"""
