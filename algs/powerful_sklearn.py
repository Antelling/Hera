from wrappers import SklearnWrapper

def gen_sklearn_powerful():
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.svm import SVR

    for regressor in [MLPRegressor(),
                      GradientBoostingRegressor(),
                      RandomForestRegressor(),
                      SVR()]:
        yield SklearnWrapper(regressor)