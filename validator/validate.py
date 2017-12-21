import data, copy, vector_math, colors
from . import score
import numpy as np


def val(algs, post, genderswap):
    people = data.get.people_raw()
    couples = data.get.couples_raw() if genderswap == False else data.get.genderswapped_couples_raw()
    y = [people[couple["female"]]["position"] for couple in couples]

    best = {
        "alg": {},
        "score": 9999,
        "post": ""
    }

    for alg in algs:
        total_score_map = {}

        for i, couple in enumerate(couples):
            new_couples = copy.deepcopy(couples)
            new_y = copy.deepcopy(y)
            del new_couples[i]
            del new_y[i]
            alg.fit(new_couples, new_y)

            maps = {
                "scoreable": {"one-way": {}},  # eg one-way, averaged, normalized
                "misc": {}  # eg mono-couples, pop-list
            }


            for person in people:
                soulmate = alg.predict([people[person]["position"]])[0]
                maps["scoreable"]["one-way"][person] = vector_math.get_rec(people, [soulmate, [1, 1, 1, 1, 1]])


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

    return best
