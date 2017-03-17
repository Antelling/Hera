class ScaleNormal(object):
    def transform(self, X):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(X)

class ScaleErf(object):
    def transform(self, X):
        import scipy.special as special
        return special.erf(X)

class ScaleJV(object):
    def transform(self, X):
        #TODO: make JV scaler
        return X

class ScalePercentile(object):
    def transform(self, X):
        #TODO: make percentile scaler
        return X