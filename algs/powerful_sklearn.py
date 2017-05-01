from wrappers import SklearnWrapper


def gen_sklearn_powerful():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.neighbors import KNeighborsRegressor

    for regressor in [
        GradientBoostingRegressor(loss="quantile"),
        SVR(C=5),
        KernelRidge(kernel="laplacian"),
        KNeighborsRegressor(n_neighbors=1)
    ]:
        for s in [False]:
            yield SklearnWrapper(regressor, scale_importance=s)
