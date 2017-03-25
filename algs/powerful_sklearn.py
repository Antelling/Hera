from wrappers import SklearnWrapper


def gen_sklearn_powerful():
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.svm import SVR

    for regressor in [RandomForestRegressor(n_estimators=60),
                          RandomForestRegressor(max_features="sqrt", n_estimators=60),
                          RandomForestRegressor(max_depth=10, n_estimators=60),
                          GradientBoostingRegressor(loss="ls"),
                          GradientBoostingRegressor(loss="lad"),
                          GradientBoostingRegressor(loss="quantile"),
                          SVR(kernel="sigmoid"),
                          SVR(C=10),
                          SVR(),
                          SVR(C=.1),
                          MLPRegressor(solver="lbfgs", alpha=0.00001),
                          MLPRegressor(solver="lbfgs"),
                          MLPRegressor(solver="lbfgs", alpha=0.00005)]:
        yield SklearnWrapper(regressor)
