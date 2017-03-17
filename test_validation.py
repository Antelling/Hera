import validator, algs, preprocessing

data_pre = [preprocessing.people.ScaleNormal(), preprocessing.people.ScaleErf()]
couples_raw_pre = [preprocessing.couples_raw.mirror()]
couples_xy_pre = [preprocessing.couples_xy.filter_outliers(contamination=.15)]

alg_gen = algs.sk_linear

validator.val(people_xy_pre=data_pre, couples_raw_pre=couples_raw_pre, couples_xy_pre=couples_xy_pre, alg_gen=alg_gen)

