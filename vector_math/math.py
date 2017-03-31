import numpy as np
import colors
import copy


def get_closest(people, point):
    # we want to produce a sorted list of [name, distance] for every person
    from sklearn.metrics.pairwise import euclidean_distances
    distances = []
    for person in people:
        person_position = people[person]["position"]
        dist = euclidean_distances([person_position], [point])[0][0]
        distances.append([person, dist])
    distances.sort(key=lambda x: x[1])
    return distances


def scale_to_relative_importance(people, importance):
    people = copy.deepcopy(people)  # stupid weird python
    for dimension_index, scale in enumerate(importance):
        for person in people:
            if scale > 1000:
                scale = 1000
            if scale < 1:
                scale = 1
            people[person]["position"][dimension_index] *= scale
    return people


def get_rec(people, soulmate):
    # okay we need to scale all the positions in people according to the relative dimension importance found in
    # soulmate[1]
    people = scale_to_relative_importance(people, soulmate[1])
    return get_closest(people, soulmate[0])


def make_relative_importance(startPos, endPos):
    vector = make_vector(startPos, endPos)
    vector = list(map(lambda x: np.abs(x), vector))
    max = np.max(vector)
    new_vector = []
    for dim in vector:
        if dim < 1:
            dim = 1
        scale = (max / dim)/2
        if scale < 1:
            scale = 1
        if scale > 5:
            scale = 5
        new_vector.append(scale)
    return new_vector


def make_vector(a, b):
    """Finds the vector difference of two points"""
    vec = []
    for i, x in enumerate(a):
        y = b[i]
        vec.append(y - x)
    return vec
