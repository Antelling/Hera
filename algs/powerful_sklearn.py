from wrappers import SklearnWrapper


def gen_sklearn_powerful():
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    for regressor in [RandomForestRegressor(n_estimators=60),
                      GradientBoostingRegressor(loss="ls"),
                      GradientBoostingRegressor(loss="lad"),
                      GradientBoostingRegressor(loss="quantile")]:
        for s in [False]:
            yield SklearnWrapper(regressor, scale_importance=s)
