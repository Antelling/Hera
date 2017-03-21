from . import get


def couples_xy(couples):
    """Takes a list of couples in raw schema and returns a sklearn style X, y tuple of starting and ending positions"""
    people = get.people_raw()
    X, y = [], []
    for couple in couples:
        X.append(people[couple["male"]]["position"])
        y.append(people[couple["female"]]["position"])
    return [X, y]


def people_xy(people):
    """takes a list of people in raw schema and returns a sklearn style X, y tuple of positions and names"""
    X, y = [], []
    for person in people:
        X.append(people[person]["position"])
        y.append(person)
    return [X, y]
