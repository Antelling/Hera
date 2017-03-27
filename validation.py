import validator, algs, preprocessing, postprocessing, sys

data_pre_options = [[preprocessing.people.Standard(), preprocessing.people.Erf()]]
couples_raw_pre_options = [[]]
from sklearn.cluster import Birch, AffinityPropagation, KMeans
couples_xy_pre_options = [[preprocessing.couples_xy.Cluster()],
                          [preprocessing.couples_xy.Cluster(Birch())],
                          [preprocessing.couples_xy.Cluster(AffinityPropagation())],
                          [preprocessing.couples_xy.Cluster(KMeans())],
                          ]

maps_post = [postprocessing.Average(),
             postprocessing.MetricEqualizer(metric="distance"),
             postprocessing.MetricEqualizer(metric="distance_median"),
             postprocessing.MetricEqualizer(metric="zscore"),
             postprocessing.MetricEqualizer(metric="zscore_median"),
             ]

things_to_test = sys.argv[1:]

best = {"score": 9999}

for data_pre in data_pre_options:
    for couples_raw_pre in couples_raw_pre_options:
        for couples_xy_pre in couples_xy_pre_options:
            if "pow" in things_to_test:
                local = validator.val(
                    people_pre=data_pre,
                    couples_raw_pre=couples_raw_pre,
                    couples_xy_pre=couples_xy_pre,
                    alg_gen=algs.sk_powerful,
                    maps_post=maps_post)

                if local["score"] < best["score"]:
                    best = local

            if "lin" in things_to_test:
                local = validator.val(
                    people_pre=data_pre,
                    couples_raw_pre=couples_raw_pre,
                    couples_xy_pre=couples_xy_pre,
                    alg_gen=algs.sk_linear,
                    maps_post=maps_post)

                if local["score"] < best["score"]:
                    best = local

            if "ker" in things_to_test:
                local = validator.val(
                    people_pre=data_pre,
                    couples_raw_pre=couples_raw_pre,
                    couples_xy_pre=couples_xy_pre,
                    alg_gen=algs.keras,
                    maps_post=maps_post)

                if local["score"] < best["score"]:
                    best = local

            if "dum" in things_to_test:
                local = validator.val(
                    people_pre=data_pre,
                    couples_raw_pre=couples_raw_pre,
                    couples_xy_pre=couples_xy_pre,
                    alg_gen=algs.dummy,
                    maps_post=maps_post)

                if local["score"] < best["score"]:
                    best = local

print("")
print(best)
