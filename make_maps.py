import preprocessing, postprocessing, data, vector_math, os, json
from wrappers import SklearnWrapper

data_pre = [preprocessing.people.Standard(), preprocessing.people.Erf()]
couples_raw_pre = [preprocessing.couples_raw.Mirror()]
couples_xy_pre = [preprocessing.couples_xy.Sanitize(contamination=.075)]

maps_post = [postprocessing.Average(),
             postprocessing.MetricEqualizer(metric="zscore_median"),
             postprocessing.JVCouples()]


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import TheilSenRegressor
alg = make_pipeline(PolynomialFeatures(3), TheilSenRegressor())
model = SklearnWrapper(alg)


people_raw = data.get.people_raw()
couples_raw = data.get.couples_raw()

for p in data_pre:
    people_raw = p.transform(people_raw)

for p in couples_raw_pre:
    couples_raw = p.transform(couples_raw)

couples_xy = data.make.couples_xy(couples_raw, people_raw)

for p in couples_xy_pre:
    couples_xy = p.transform(couples_xy)


print('training model')
model.fit(*couples_xy)
print("done")
print("")


maps = {
    "scoreable": {"one-way": {}},  # eg one-way, averaged, normalized
    "misc": {}  # eg mono-couples, pop-list
}

print("making one-way")
for person in people_raw:
    soulmate = model.predict_for_single_point(people_raw[person])
    maps["scoreable"]["one-way"][person] = vector_math.get_rec(people_raw, soulmate)
print('done')
print("")

for processor in maps_post:
    maps = processor.transform(maps)


people_in_relationships = []
for couple in data.get.couples_raw():
    if couple["still_dating"]:
        people_in_relationships.append(couple["male"])
        people_in_relationships.append(couple["female"])


map_to_save = {
    "map": maps["scoreable"]["ZScoreMedianNormalized"],
    "couples": maps["misc"]["JVCouples"],
    "people_raw": people_raw,
    "people_in_relationships": people_in_relationships
}


this_file = os.path.dirname(os.path.realpath(__file__))
target = os.path.join(this_file, "display_server", "static", "map.json")
data = json.dumps(map_to_save)
open(target, "w").write(data)