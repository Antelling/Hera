from data import make, get
from abc import ABC
import colors, copy
from sklearn.base import BaseEstimator


class TransformerBase(ABC, BaseEstimator):
    def fit(self, *_):
        return self

    def transform(self, X):
        pass


class AddCouplesXy(TransformerBase):
    def transform(self, data):
        if "couples_raw" in data and "female" in data["couples_raw"][0]:
            couples = data["couples_raw"]
            people = data["people"]
            X = []
            y = []
            for couple in couples:
                X.append(people[couple["male"]]["position"])
                y.append(people[couple["female"]]["y"])
            data["couples_xy"] = [X, y]
        return data


class GetXy(TransformerBase):
    def transform(self, data):
        if "couples_xy" in data:
            return data["couples_xy"]
        else:
            people = list(data["people"].items())
            people.sort(key=lambda x: int(x[0]))
            positions = []
            for person in people:
                positions.append(person[1]["position"])
            return positions


class GetCouplesRaw(TransformerBase):
    def transform(self, data):
        return data["couples_raw"]


class GetPeople(TransformerBase):
    def transform(self, data):
        return data["people"]


class GetPeoplesXy(TransformerBase):
    def transform(self, data):
        return make.people_xy(data["people"])


class FormData(TransformerBase):
    def __init__(self, alg):
        self.alg = alg

    def transform(self, couplesRaw):
        if type(couplesRaw[0]) == dict:
            d = {"people": get.people_decomposed(self.alg), "couples_raw": couplesRaw}
            return d
        else:
            # we are passed a list of positions, so we need to search through people to find original positions
            people = get.people_decomposed(self.alg)
            new_positions = []
            for position in couplesRaw:
                for person in people:
                    if people[person]['y'] == position:
                        new_positions.append(people[person]["position"])
                        break
            d = {"people": {}, "couples_raw": []}
            for i, position in enumerate(new_positions):
                d["people"][str(i)] = {"position": position,
                                       'y': copy.deepcopy(position)}  # so we add a fake person to hold position
                d["couples_raw"].append(
                    {'male': str(i)})  # and a fake relationship. The female position will be predicted.
            return d


class Pass(TransformerBase):
    def __init__(self, contamination=None, clusterer=None, replace=None, alg=None):
        #self.contamination = contamination
        #self.clusterer = KMeans()
        pass

    def transform(self, X):
        return X
