from . import get


def couples_xy(couples, people=None):
    """Takes a list of couples in raw schema and returns a sklearn style X, y tuple of starting and ending positions"""
    if people is None:
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


def remove_creepy_age_gap(name, dist_list):
    people_raw = get.people_raw()
    new_list = []
    grade = people_raw[name]["grade"]
    for otherperson in dist_list:
        if is_okay(grade, people_raw[otherperson[0]]["grade"]):
            new_list.append(otherperson)
    return new_list


def is_okay(grade1, grade2):
    if grade1 == 9:
        okay_list = [9, 10]
    if grade1 == 10:
        okay_list = [9, 10, 11, 12]
    if grade1 == 11:
        okay_list = [10, 11, 12]
    if grade1 == 12:
        okay_list = [10, 11, 12, 13]
    if grade1 == 13:
        okay_list = [12, 13]
    if grade1 == 14:
        okay_list = [14]
    return grade2 in okay_list
