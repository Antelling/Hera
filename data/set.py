"""Provides a method of saving new people to people.json. Couples are to be saved manually be editing the json file."""

import json, os

file_path = os.path.dirname(os.path.realpath(__file__))


def save_person(*, name, gender, grade, display, group, position):
    """saves the passed person to people.json"""
    assert type(name) is str
    assert type(group) is str
    assert gender in ["male", "female"]
    assert type(grade) is int
    assert type(display) is bool
    assert type(position) is list

    position = list(map(lambda x: int(x), position))

    file_name = os.path.join(file_path, "people.json")
    file = open(file_name)
    data = file.read()
    file.close()
    data = json.loads(data)
    data[name] = {
        "group": group,
        "gender": gender,
        "grade": grade,
        "display": display,
        "position": position
    }
    file = open(file_name, "w")
    file.write(json.dumps(data))
    file.flush()
    file.close()
