import data, copy, vector_math, colors
from . import score
import numpy as np


def val(*, people_pre, couples_raw_pre, couples_xy_pre, alg_gen, maps_post):
    people_raw = data.get.people_raw()
    for processor in people_pre:
        people_raw = processor.transform(people_raw)

    couples_raw = data.get.couples_raw()

    best = {
        "alg": {},
        "score": 9999,
        "post": ""
    }

    for alg in alg_gen():
        try:
            total_score_map = {}

            for i, couple in enumerate(couples_raw):
                new_couples = copy.deepcopy(couples_raw)
                del new_couples[i]

                for processor in couples_raw_pre:
                    new_couples = processor.transform(new_couples)
                couples_xy = data.make.couples_xy(new_couples, people_raw)
                for processor in couples_xy_pre:
                    couples_xy = processor.transform(couples_xy)

                alg.fit(*couples_xy)

                maps = {
                    "scoreable": {"one-way": {}},  # eg one-way, averaged, normalized
                    "misc": {}  # eg mono-couples, pop-list
                }

                for person in people_raw:
                    soulmate = alg.predict_for_single_point(people_raw[person])
                    maps["scoreable"]["one-way"][person] = vector_math.get_rec(people_raw, soulmate)

                for processor in maps_post:
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
                    best["pre"] = [people_pre, couples_raw_pre, couples_xy_pre]
        except Exception as e:
            colors.red(e)
            KeyboardInterrupt("roflmao")

    return best