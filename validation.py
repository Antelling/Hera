import validator, algs, preprocessing, postprocessing, sys, colors

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn #shut up sklearn

from sklearn.manifold import TSNE

data_pre_options = [
    [preprocessing.people.Decompose(TSNE(n_components=4))],
    [preprocessing.people.Decompose(TSNE(n_components=3))],
    [preprocessing.people.Flatten(), preprocessing.people.Standard()],
    [preprocessing.people.Standard(), preprocessing.people.Erf()]
]

couples_raw_pre_options = [
    [preprocessing.couples_raw.Mirror()],
    [preprocessing.couples_raw.Mirror(), preprocessing.couples_raw.PositionFiltering(max=.66)],
    [preprocessing.couples_raw.Mirror(), preprocessing.couples_raw.PositionFiltering(max=.3)],
    [],
]


from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
couples_xy_pre_options = [
    [],
    [preprocessing.couples_xy.SanitizeStartVec()],
    [preprocessing.couples_xy.Cluster(SpectralClustering(n_clusters=15))],
    [preprocessing.couples_xy.SanitizeStartVec(), preprocessing.couples_xy.Cluster(SpectralClustering(n_clusters=15))],
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
    for couples_raw_pre in couples_raw_pre_options:
        for couples_xy_pre in couples_xy_pre_options:
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