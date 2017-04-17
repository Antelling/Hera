from wrappers import SklearnWrapper


def gen_sklearn_linear():
    polynomial_degrees = [1, 4, 3]

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression, HuberRegressor

    for degree in polynomial_degrees:
        for regressor in [
            LinearRegression(),
            HuberRegressor()
        ]:
            for s in [False]:
                regressor = make_pipeline(PolynomialFeatures(degree), regressor)
                regressor = SklearnWrapper(regressor, scale_importance=s)
                yield regressor
