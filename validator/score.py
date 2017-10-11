import numpy as np
import data


def find_pos(distances, target):
    people_raw = data.get.people_raw() #we use this for genders only
    target_gen = people_raw[target]["gender"]
    i = 0
    for distance in distances:
        if target_gen != people_raw[distance[0]]["gender"]: #we only worry about the opposite gender when penalizing
            continue
        i += 1
        if distance[0] == target:
            return i


def score(maps, couple):
    score_map = {}
    for scoreable in maps["scoreable"]:
        distances = []
        distances.append(find_pos(maps["scoreable"][scoreable][couple["male"]], couple["female"]))
        score_map[scoreable] = list(filter(lambda x: x is not None, distances))
    return score_map
