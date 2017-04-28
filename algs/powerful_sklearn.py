from wrappers import SklearnWrapper


def gen_sklearn_powerful():
    from sklearn.ensemble import GradientBoostingRegressor

    for regressor in [
        GradientBoostingRegressor(loss="quantile"),
        GradientBoostingRegressor(loss="quantile", alpha=.7),
        GradientBoostingRegressor(loss="quantile", alpha=.95),
        GradientBoostingRegressor(loss="lad"),
    ]:
        for s in [False]:
            yield SklearnWrapper(regressor, scale_importance=s)
