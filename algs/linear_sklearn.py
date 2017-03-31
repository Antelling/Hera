from wrappers import SklearnWrapper


def gen_sklearn_linear():
    polynomial_degrees = [1, 2, 3]

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression, TheilSenRegressor, HuberRegressor, RANSACRegressor

    for degree in polynomial_degrees:
        for regressor in [LinearRegression(),
                          TheilSenRegressor(),
                          HuberRegressor(),
                          RANSACRegressor()]:
            for s in [True, False]:
                regressor = make_pipeline(PolynomialFeatures(degree), regressor)
                regressor = SklearnWrapper(regressor, scale_importance=s)
                yield regressor
