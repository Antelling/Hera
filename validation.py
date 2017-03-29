import validator, algs, preprocessing, postprocessing, sys, colors

data_pre_options = [[preprocessing.people.Standard(), preprocessing.people.Erf()]]
couples_raw_pre_options = [
    [],
    [
        preprocessing.couples_raw.Mirror()
    ],
    [preprocessing.couples_raw.Time_mod()],
    [preprocessing.couples_raw.Time_mod(mod=lambda x:x*x)],
    [preprocessing.couples_raw.Time_mod(mod=lambda x:x)],
    [
        preprocessing.couples_raw.Mirror(),
        preprocessing.couples_raw.Time_mod()
    ]]
from sklearn.cluster import SpectralClustering

couples_xy_pre_options = [
    [preprocessing.couples_xy.Cluster(SpectralClustering(n_clusters=13))],
    [preprocessing.couples_xy.Cluster(SpectralClustering(n_clusters=17))],
    [preprocessing.couples_xy.Cluster(SpectralClustering(n_clusters=22))],
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
                    colors.purple(local)
                    colors.green(local["score"])
                    best = local

            if "lin" in things_to_test:
                local = validator.val(
                    people_pre=data_pre,
                    couples_raw_pre=couples_raw_pre,
                    couples_xy_pre=couples_xy_pre,
                    alg_gen=algs.sk_linear,
                    maps_post=maps_post)

                if local["score"] < best["score"]:
                    colors.purple(local)
                    colors.green(local["score"])
                    best = local

            if "ker" in things_to_test:
                local = validator.val(
                    people_pre=data_pre,
                    couples_raw_pre=couples_raw_pre,
                    couples_xy_pre=couples_xy_pre,
                    alg_gen=algs.keras,
                    maps_post=maps_post)

                if local["score"] < best["score"]:
                    colors.purple(local)
                    colors.green(local["score"])
                    best = local

            if "dum" in things_to_test:
                local = validator.val(
                    people_pre=data_pre,
                    couples_raw_pre=couples_raw_pre,
                    couples_xy_pre=couples_xy_pre,
                    alg_gen=algs.dummy,
                    maps_post=maps_post)

                if local["score"] < best["score"]:
                    colors.purple(local)
                    colors.green(local["score"])
                    best = local

print("")
print(best)
