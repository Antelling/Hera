"""
Collection of functions for loading data. This attempts to be as efficient on IO and CPU as possible, forming a cache
of items, but only when those items are needed. If a certain function is never called, the object it returns will
never be loaded.
"""

import json, os
from . import make

file_path = os.path.dirname(os.path.realpath(__file__))

cache = {}


def people_raw():
    """Return a list of people in the following schema:
        gender: str/cat, male|female
        grade: int/cat, 9|10|11|12|13..., 13 is freshmen year of college
        display: bool, if false, never ever display them. Acts kind of as a failsafe for certain people
        group: str/cat, boiling springs|carlisle|cv|big springs|internet..., their "region"
        position: list{int}, a list of their IPIP 5 dimensions"""
    if "people_raw" in cache:
        return cache["people_raw"]
    else:
        cache["people_raw"] = json.loads(open(os.path.join(file_path, "people.json"), "r").read())
        return cache["people_raw"]


def people_xy():
    """Return a (X,y) tuple where X is a list of IPIP 5 dimensions, and y is the name"""
    if "people_xy" in cache:
        return cache["people_xy"]
    else:
        people = people_raw()
        cache["people_xy"] = make.people_xy(people)
        return cache["people_xy"]


def couples_raw():
    """Return a list of couples defined by the schema:
        male: str/key, The name of the male person
        female: str/key, The name of the female person
        length: int, The amount of time they have been dating
        orientation: str/cat, straight|gay|lesbian, if gay or lesbian, male or female contains a female or male, respectively
        still_dating: bool, are they still dating or have they broken up
    """
    if "couples_raw" in cache:
        return cache["couples_raw"]
    else:
        cache["couples_raw"] = json.loads(open(os.path.join(file_path, "couples.json"), "r").read())
        return cache["couples_raw"]


def couples_xy():
    """returns an X, y list where X is male positions and y is female positions"""
    if "couples_xy" in cache:
        return cache["couples_xy"]
    else:
        couples = couples_raw()
        cache["couples_xy"] = make.couples_xy(couples)
        return cache["couples_xy"]


def couples_list():
    """return list of people in relationships"""
    if "couples_list" in cache:
        return cache["couples_list"]
    else:
        l = []
        for couple in couples_raw():
            l.append(couple["male"])
            l.append(couple["female"])
        cache["couples_list"] = l
        return l

def couples_pairs():
    if "couples_pairs" in cache:
        return cache["couples_pairs"]
    else:
        pairs = {}
        for couple in couples_raw():
            pairs[couple["male"]] = couple["female"]
            pairs[couple["female"]] = couple["male"]
        cache["couples_pairs"] = pairs
        return pairs
