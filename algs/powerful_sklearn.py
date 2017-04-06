from wrappers import SklearnWrapper


def gen_sklearn_powerful():
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR

    for regressor in [
        GradientBoostingRegressor(loss="quantile"),
        RandomForestRegressor(),
        MLPRegressor(),
        SVR()
    ]:
        for s in [False]:
            yield SklearnWrapper(regressor, scale_importance=s)
