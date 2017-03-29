import numpy as np

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
    for dimension_index, scale in enumerate(importance):
        for person in people:
            people[person]["position"][dimension_index] *= scale
    return people


def get_rec(people, soulmate):
    # okay we need to scale all the positions in people according to the relative dimension importance found in
    # soulmate[1]
    people = scale_to_relative_importance(people, soulmate[1])
    return get_closest(people, soulmate[0])


def make_relative_importance(startPos, endPos):
    vector = make_vector(startPos, endPos)
    vector = list(map(lambda x:np.abs(x), vector))
    max = np.max(vector)
    vector = list(map(lambda x:max/x, vector))
    return vector


def make_vector(a, b):
    """Finds the vector difference of two points"""
    vec = []
    for i, x in enumerate(a):
        y = b[i]
        vec.append(y - x)
    return vec