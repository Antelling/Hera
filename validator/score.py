import numpy as np
import data


def find_pos(distances, target):
    people_raw = data.get.people_raw() #use this for genders only
    target_gen = people_raw[target]["gender"]
    i = 0
    for distance in distances:
        if target_gen != people_raw[distance[0]]["gender"]:
            continue
        i += 1
        if distance[0] == target:
            return i


def score(maps, couple):
    score_map = {}
    for scoreable in maps["scoreable"]:
        distances = []
        for pair in [[couple["male"], couple["female"]], [couple["female"], couple["male"]]]:
            distances.append(find_pos(maps["scoreable"][scoreable][pair[0]], pair[1]))
        score_map[scoreable] = distances
    return score_map
