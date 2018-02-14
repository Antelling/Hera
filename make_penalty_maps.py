import postprocessing, data, vector_math, os, json, random
from sklearn.externals import joblib
import numpy as np

info = []
for file in os.listdir("penalty_model_info"):
    d = json.loads(open(os.path.join("penalty_model_info", file)).read())
    name = d["name"]
    d = np.mean(d[name])
    info.append([name, d])

info.sort(key=lambda x:x[1])

print(info)

model = joblib.load(os.path.join("penalty_models", info[0][0]))

#now we retrain our models on up-to-date couple info

#region gen data
def gen_penalty(months):
    if months > 23: return 1
    if months > 3: return 2
    return 3


couples = data.get.couples_raw()
people = data.get.people_raw()
people_xy, names = data.get.people_xy()

for i, couple in enumerate(couples):
    penalty = 0 if couple["married"] == True else gen_penalty(couple["length"])
    couple["penalty"] = penalty
    couple[i] = couple


def gen_training_data(couples):
    X, y = [], []

    for couple in couples:
        x = people[couple["male"]]["position"] + people[couple["female"]]["position"]
        X.append(x)
        y.append(couple["penalty"])

    # now we need some negative examples
    for _ in range(len(couples) * 7):
        m = random.choice(people_xy)
        f = random.choice(people_xy)
        vector = m + f
        if vector not in X:
            X.append(vector)
            y.append(6)
    return X, y

#endregion


X, y = gen_training_data(couples)


model.fit(X, y)

maps = {
    "scoreable": {"average": {}},  # we can't have one-way maps with this strategy
    "misc": {}  # eg mono-couples, pop-list
}

for name in people:
    person = people[name]
    penalties = []
    for othername in people:
        otherperson = people[othername]
        vec = person["position"] + otherperson["position"]
        pen = model.predict([vec])[0].tolist()
        penalties.append([othername, pen])
    penalties.sort(key=lambda x: x[1])
    maps["scoreable"]["average"][name] = penalties


# okay so now we have a recommendation for every person
# but if someone is in a relationship, they'll just get the person they are dating as like their first result
# which isn't necessarily a bad thing, since it makes this look accurate, but I'm using broken up couples in my dataset,
# and people get upset when I tell them to date their ex. So:
print("normal recs made, making recs for people in couples: ")
import copy
for i, couple in enumerate(couples):
    if not couple["still_dating"]:
        new_X = copy.deepcopy(X)
        new_y = copy.deepcopy(y)
        model.fit(new_X, new_y)

        for gender_key in ["male", "female"]:
            name = couple[gender_key]
            person = people[name]
            penalties = []
            for othername in people:
                otherperson = people[othername]
                vec = person["position"] + otherperson["position"]
                pen = model.predict([vec])[0].tolist()
                penalties.append([othername, pen])
            penalties.sort(key=lambda x: x[1])
            maps["scoreable"]["average"][name] = penalties

print("")
print("done")

maps_post = [postprocessing.MetricEqualizer(metric="percentage", name="main"),
             postprocessing.LeastCompatible(),
             postprocessing.JVCouples()]

for processor in maps_post:
    maps = processor.transform(maps)

people_in_relationships = []
for couple in data.get.couples_raw():
    if couple["still_dating"]:
        people_in_relationships.append(couple["male"])
        people_in_relationships.append(couple["female"])

map_to_save = {
    "map": maps["scoreable"]["main"],
    "couples": maps["misc"]["JVCouples"],
    "people_raw": people,
    "list": maps["misc"]["list"],
    "people_in_relationships": people_in_relationships
}

this_file = os.path.dirname(os.path.realpath(__file__))
target = os.path.join(this_file, "display_server", "static", "map.json")
data = json.dumps(map_to_save, indent=4)
open(target, "w").write(data)
