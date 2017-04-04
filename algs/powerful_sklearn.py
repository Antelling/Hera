from wrappers import SklearnWrapper


def gen_sklearn_powerful():
    from sklearn.ensemble import GradientBoostingRegressor

    for regressor in [
        GradientBoostingRegressor(loss="ls"),
        GradientBoostingRegressor(loss="lad"),
        GradientBoostingRegressor(loss="quantile"),
        GradientBoostingRegressor(loss="huber")
    ]:
        for s in [False]:
            yield SklearnWrapper(regressor, scale_importance=s)
