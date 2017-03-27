import preprocessing, postprocessing, data, vector_math, os, json
from wrappers import SklearnWrapper

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import TheilSenRegressor

alg = make_pipeline(PolynomialFeatures(1), TheilSenRegressor())
model = SklearnWrapper(alg)

people_raw = data.get.people_raw()
data_pre = [preprocessing.people.Standard(), preprocessing.people.Erf()]
for p in data_pre:
    people_raw = p.transform(people_raw)

couples_raw = data.get.couples_raw()
couples_raw_pre = []
from sklearn.cluster import SpectralClustering
couples_xy_pre = [preprocessing.couples_xy.Cluster(SpectralClustering(n_clusters=12))]


def make_xy(people, couples, raw_trans, xy_trans):
    for p in raw_trans:
        couples = p.transform(couples)
    couples = data.make.couples_xy(couples, people)
    for p in xy_trans:
        couples = p.transform(couples)
    return couples


couples_xy = make_xy(people_raw, couples_raw, couples_raw_pre, couples_xy_pre)
model.fit(*couples_xy)

maps = {
    "scoreable": {"one-way": {}},  # eg one-way, averaged, normalized
    "misc": {}  # eg mono-couples, pop-list
}

for person in people_raw:
    soulmate = model.predict_for_single_point(people_raw[person])
    maps["scoreable"]["one-way"][person] = vector_math.get_rec(people_raw, soulmate)

# okay so now we have a recommendation for every person
# but if someone is in a relationship, they'll just get the person they are dating as like their first result
# which isn't necessarily a bad thing, since it makes this look accurate, but I'm using broken up couples in my dataset,
# and people get upset when I tell them to date their ex. So:
print("normal recs made, making recs for people in couples: ")
import copy
for i, couple in enumerate(couples_raw):
    new_couples = copy.deepcopy(couples_raw)
    del new_couples[i]
    couples_xy = make_xy(people_raw, new_couples, couples_raw_pre, couples_xy_pre)
    model.fit(*couples_xy)
    print(".", end="")
    for gender_key in ["male", "female"]:
        person = couple[gender_key]
        soulmate = model.predict_for_single_point(people_raw[person])
        maps["scoreable"]["one-way"][person] = vector_math.get_rec(people_raw, soulmate)
print("")
print("done")

maps_post = [postprocessing.Average(),
             postprocessing.MetricEqualizer(metric="distance_median", name="main"),
             postprocessing.JVCouples(),
             postprocessing.OrderedRecs()]

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
    "list": maps["misc"]["OrderedRecs"],
    "people_raw": people_raw,
    "people_in_relationships": people_in_relationships
}

this_file = os.path.dirname(os.path.realpath(__file__))
target = os.path.join(this_file, "display_server", "static", "map.json")
data = json.dumps(map_to_save)
open(target, "w").write(data)
