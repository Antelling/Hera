from wrappers import SklearnWrapper


def gen_sklearn_powerful():
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.svm import SVR

    for regressor in [GradientBoostingRegressor(),
                      GradientBoostingRegressor(loss="lad"),
                      GradientBoostingRegressor(loss="quantile"),
                      RandomForestRegressor(),
                      SVR(),
                      MLPRegressor()]:
        yield SklearnWrapper(regressor)
