import postprocessing, data, vector_math, os, json
from sklearn.externals import joblib
import numpy as np


info = json.loads(open("model_info.json").read())

m2f = []
f2m = []

for model in info:
    m2f.append((model["name"], np.mean(model["m2f"])))
    f2m.append((model["name"], np.mean(model["f2m"])))

m2f.sort(key=lambda x:x[1])
f2m.sort(key=lambda x:x[1])

print(f2m[0][1])
print(m2f[0][1])

female_model = joblib.load(os.path.join("models", f2m[0][0]))
male_model = joblib.load(os.path.join("models", m2f[0][0]))
models = [male_model, female_model]


#now we retrain our models on up-to-date couple info
normal = data.get.couples_raw()
swapped = data.get.genderswapped_couples_raw()
people = data.get.people_raw()
for m_index, couple_data in (0, normal), (1, swapped):
    y = [people[couple["female"]]["position"] for couple in couple_data]
    models[m_index].fit(couple_data, y)


maps = {
    "scoreable": {"one-way": {}},  # eg one-way, averaged, normalized
    "misc": {}  # eg mono-couples, pop-list
}

for person in people:
    model = models[0] if people[person]["gender"] == "male" else models[1]
    soulmate = model.predict([people[person]["position"]])[0]
    maps["scoreable"]["one-way"][person] = vector_math.get_rec(people, [soulmate, [1, 1, 1, 1, 1]])


# okay so now we have a recommendation for every person
# but if someone is in a relationship, they'll just get the person they are dating as like their first result
# which isn't necessarily a bad thing, since it makes this look accurate, but I'm using broken up couples in my dataset,
# and people get upset when I tell them to date their ex. So:
print("normal recs made, making recs for people in couples: ")
import copy
for i, couple in enumerate(normal):
    if not couple["still_dating"]:
        new_normal = copy.deepcopy(normal)
        new_swapped = copy.deepcopy(swapped)
        new_normal_y = [people[couple["female"]]["position"] for couple in copy.deepcopy(new_normal)]
        new_swapped_y = [people[couple["female"]]["position"] for couple in copy.deepcopy(new_swapped)]
        del new_normal[i]
        del new_swapped[i]
        del new_normal_y[i]
        del new_swapped_y[i]
        models[0].fit(new_normal, new_normal_y)
        models[1].fit(new_swapped, new_swapped_y)

        print(str(i + 1) + "/" + str(len(normal)))

        for gender_key in ["male", "female"]:
            person = couple[gender_key]
            model = models[0] if gender_key == "male" else models[1]
            soulmate = model.predict([people[person]["position"]])[0]
            maps["scoreable"]["one-way"][person] = vector_math.get_rec(people, [soulmate, [1, 1, 1, 1, 1]])
print("")
print("done")

maps_post = [postprocessing.Average(),
             postprocessing.MetricEqualizer(metric="percentage", name="main"),
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
