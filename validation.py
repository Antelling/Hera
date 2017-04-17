import validator, algs, preprocessing, postprocessing, sys, colors

from sklearn.manifold import TSNE

data_pre_options = [
    [
        preprocessing.people.Decompose(TSNE(n_components=4)),
        preprocessing.people.Standard()],
    [
        preprocessing.people.Decompose(TSNE(n_components=4)),
        preprocessing.people.Flatten(),
        preprocessing.people.Standard()
    ],
]

couples_raw_pre_options = [
    [preprocessing.couples_raw.Mirror()],
    [preprocessing.couples_raw.Mirror(), preprocessing.couples_raw.PositionFiltering(.7)],
    [preprocessing.couples_raw.Mirror(), preprocessing.couples_raw.PositionFiltering(.4)],
]

from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans

couples_xy_pre_options = [
    [],
]

maps_post = [
    postprocessing.Average(),
    postprocessing.CoupleEqualizerFast(),
    postprocessing.MetricEqualizer(metric="distance"),
    postprocessing.MetricEqualizer(metric="zscore")
]

things_to_test = sys.argv[1:]

best = {"score": 9999}

for data_pre in data_pre_options:
    colors.purple("1")
    for couples_raw_pre in couples_raw_pre_options:
        colors.blue("2")
        for couples_xy_pre in couples_xy_pre_options:
            colors.green("3")
            if "pow" in things_to_test:
                local = validator.val(
                    people_pre=data_pre,
                    couples_raw_pre=couples_raw_pre,
                    couples_xy_pre=couples_xy_pre,
                    alg_gen=algs.sk_powerful(),
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
                    alg_gen=algs.sk_linear(),
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
                    alg_gen=algs.keras(),
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
                    alg_gen=algs.dummy(),
                    maps_post=maps_post)

                if local["score"] < best["score"]:
                    colors.purple(local)
                    colors.green(local["score"])
                    best = local

            if "nai" in things_to_test:
                local = validator.val(
                    people_pre=data_pre,
                    couples_raw_pre=couples_raw_pre,
                    couples_xy_pre=couples_xy_pre,
                    alg_gen=algs.naive(),
                    maps_post=maps_post)

                if local["score"] < best["score"]:
                    colors.purple(local)
                    colors.green(local["score"])
                    best = local

print("")
print(best)
