from sklearn.preprocessing import StandardScaler

class ScaleNormal(object):
    def transform(self, X):
        scaler = StandardScaler()
        X = scaler.fit_transform(X).tolist()
        return X

class ScaleErf(object):
    def transform(self, X):
        import scipy.special as special
        return special.erf(X).tolist()

class ScaleJV(object):
    def transform(self, X):
        #TODO: make JV scaler
        return X

class ScalePercentile(object):
    def transform(self, X):
        #TODO: make percentile scaler
        return X