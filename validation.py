import validator, algs, preprocessing, postprocessing

data_pre = [preprocessing.people.ScaleNormal(), preprocessing.people.ScaleErf()]
couples_raw_pre = [preprocessing.couples_raw.Mirror()]
couples_xy_pre = [preprocessing.couples_xy.Sanitize(contamination=.075)]

maps_post = [postprocessing.Average(),
             postprocessing.JVCouples()
             ]

alg_gen = algs.sk_powerful

validator.val(
    people_xy_pre=data_pre,
    couples_raw_pre=couples_raw_pre,
    couples_xy_pre=couples_xy_pre,
    alg_gen=alg_gen,
    maps_post=maps_post)
