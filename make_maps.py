import postprocessing, data, vector_math, os, json
from sklearn.externals import joblib


model = joblib.load("model.pkl")

couples = data.get.couples_raw()
people = data.get.people_raw()
y = [people[couple["female"]]["position"] for couple in couples]

model.fit(couples, y)


maps = {
    "scoreable": {"one-way": {}},  # eg one-way, averaged, normalized
    "misc": {}  # eg mono-couples, pop-list
}

for person in people:
    soulmate = model.predict([people[person]["position"]])[0]
    maps["scoreable"]["one-way"][person] = vector_math.get_rec(people, [soulmate, [1, 1, 1, 1, 1]])


# okay so now we have a recommendation for every person
# but if someone is in a relationship, they'll just get the person they are dating as like their first result
# which isn't necessarily a bad thing, since it makes this look accurate, but I'm using broken up couples in my dataset,
# and people get upset when I tell them to date their ex. So:
print("normal recs made, making recs for people in couples: ")
import copy
for i, couple in enumerate(couples):
    if not couple["still_dating"]:
        new_couples = copy.deepcopy(couples)
        new_y = copy.deepcopy(y)
        del new_couples[i]
        del new_y[i]
        model.fit(new_couples, new_y)
        print(str(i + 1) + "/" + str(len(couple)))
        for gender_key in ["male", "female"]:
            person = couple[gender_key]
            soulmate = model.predict([people[person]["position"]])[0]
            maps["scoreable"]["one-way"][person] = vector_math.get_rec(people, [soulmate, [1, 1, 1, 1, 1]])
print("")
print("done")

maps_post = [postprocessing.Average(name="main"),
             #.MetricEqualizer(metric="percentage", name="main"),
             #postprocessing.RedBadCouples(),
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
    #"list": maps["misc"]["list"],
    "people_in_relationships": people_in_relationships
}

this_file = os.path.dirname(os.path.realpath(__file__))
target = os.path.join(this_file, "display_server", "static", "map.json")
data = json.dumps(map_to_save, indent=4)
open(target, "w").write(data)
