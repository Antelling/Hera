import colors, data, random
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from pipeline import pipeline, param_grids

from sklearn.model_selection import RandomizedSearchCV

def dist_score(estimator, X, y):
    """Calculate residuals of estimator"""

    # okay so the pipeline object will do a lot of mirroring and filtering to couple objects
    # which we don't want to happen, we want to get just the predictions
    # so we extract positions and pass those instead, which instructs the internal pipeline handlers to skip the
    # couple transformer steps
    people = data.get.people_raw()
    positions = [people[couple["male"]]["position"] for couple in X]
    predictions = estimator.predict(positions)
    residuals = euclidean_distances(predictions, y)
    return -np.mean(residuals)

gender_swap = [True, False]
random.shuffle(gender_swap)
while True:
    for swap in gender_swap:

        best_score = 10000
        best_cv = {}

        #I've kind of hardcoded male -> female in all my code
        #so let's just change the place of male and female here
        #since otherwise I would have to like pass through the gender direction in the pipeline and ew
        couples = data.get.couples_raw() if swap == False else data.get.genderswapped_couples_raw()

        people = data.get.people_raw()
        y = [people[couple["female"]]["position"] for couple in couples]

        from sklearn.externals import joblib

        scores = {}
        for name, grid in param_grids:
            scores[name] = []

        from validator import val
        from postprocessing import MetricEqualizer, Average

        # we perform a 7-fold validated random search over 7 param options 7 times, then switch genders: B I B L I C A L
        # no but for real the exact numbers don't really matter
        for _ in range(7):
            for name, param_grid in param_grids:
                try:
                    rand_cv = RandomizedSearchCV(
                        pipeline,
                        param_distributions=param_grid,
                        n_iter=7,
                        scoring=dist_score,
                        cv=7,
                        return_train_score=False
                    )
                    rand_cv.fit(couples, y)

                    score = val([rand_cv.best_estimator_], [Average(), MetricEqualizer(metric="percentage")])["score"]
                    print(score)

                    scores[name].append(score)

                    if score < best_score:
                        best_score = score
                        best_cv = rand_cv
                except Exception as e:
                    colors.red(e)

            model_name = "m2f" if gender_swap == False else "f2m"
            joblib.dump(best_cv.best_estimator_, "models/" + str(round(best_score, 2)) + model_name + ".pkl")

            colors.white("_________________________")
            for name in scores:
                colors.blue(name + ": " + str(np.mean(scores[name])))
            colors.green(best_score)
            colors.white(best_cv.best_estimator_.get_params())
