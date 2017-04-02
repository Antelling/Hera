from sklearn.preprocessing import StandardScaler
import data
import numpy as np
import random

class PeopleBase(object):
    def transform(self, people):
        people_xy = data.make.people_xy(people)
        people_xy[0] = self.transform_x(people_xy[0])
        for i, person in enumerate(people):
            people[person]["position"] = people_xy[0][i]
        return people

    def transform_x(self, X):
        return X

class Standard(PeopleBase):
    def transform_x(self, X):
        scaler = StandardScaler()
        X = scaler.fit_transform(X).tolist()
        return X


class Erf(PeopleBase):
    def transform_x(self, X):
        import scipy.special as special
        return special.erf(X).tolist()


class Flatten(PeopleBase):
    def transform_x(self, X):
        X = np.array(X)
        X = np.rot90(X, k=1)
        X = list(map(self.flatten_dimension, X))
        X = np.rot90(X, k=3)
        return X.tolist()

    def flatten_dimension(self, d):
        n = len(d)
        step = 100 / n
        numbers = np.arange(0, 100, step)

        #okay so we want to snap our d to numbers
        #but we need to remember our original position
        #and we need to fuzz the numbers to avoid bad correlations
        structured_d = []
        for i, value in enumerate(d):
            structured_d.append({"index": i, "value": value + random.uniform(-.4, .4)})

        structured_d.sort(key=lambda x:x["value"])

        for i, value in enumerate(structured_d):
            structured_d[i]["flattened_value"] = numbers[i]

        structured_d.sort(key=lambda x:x["index"])

        d = []
        for value in structured_d:
            d.append(value["flattened_value"])
        return d