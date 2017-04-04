import data, wrappers, colors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import preprocessing

from sklearn.metrics.pairwise import euclidean_distances

import random
import numpy as np


class ANTSAC():
    def __init__(self, evaporation=.9, min_inliers=99999, max_iter=75, threshold=None):
        self.evaporation = evaporation
        self.min_inliers = min_inliers
        self.max_iter = max_iter
        self.threshold = threshold

    def pheromone_delta(self, points, inliers_len, mean_previous_inliers, generations, residual, theta):
        first_part = (inliers_len / (len(points)) + (mean_previous_inliers / generations))
        second_part = np.exp(-.5 * ((residual / theta) ** 2))
        return first_part * second_part

    def select_points(self, probability_list, points):
        k = len(points) - 10
        # okay so because we are using tuples of lists, numpy considers this to be a 3d array
        # which I guess it technically is, but we want to treat it as a 1D

        prob_sum = np.sum(probability_list)
        probability_list = [p/prob_sum for p in probability_list]

        indexes = list(range(len(points)))
        indexes = np.random.choice(indexes, size=(k,), p=probability_list, replace=False)
        chosen_points = [points[x] for x in indexes]
        indexes.sort()
        colors.green(indexes)
        return chosen_points, indexes

    def ANTSAC(self, points):
        alg = wrappers.SklearnWrapper(LinearRegression(), unidirectional=True)

        best = {"inliers": 0}
        pheromone_matrix = []
        for _ in points:
            pheromone_matrix.append(1 / len(points))
        pheromone_matrix = [pheromone_matrix]

        X, y = list(zip(*points))

        if self.threshold is None:
            self.threshold = self.MAD(y) * 5

        inliers_acheived_so_far = []

        for _ in range(self.max_iter):
            subset, indexes = self.select_points(
                pheromone_matrix[-1],
                points)  # select random points according to the pheromone probability distribution
            subset_X, subset_y = list(zip(*subset))  # turn the list of tuples into two tuples of lists
            subset_X, subset_y = list(subset_X), list(subset_y)  # turn the tuples into lists

            alg.fit(subset_X, subset_y)
            predictions = alg.predict(X)
            predictions = [x[0] for x in predictions]
            residuals = self.calc_resids(y, predictions)
            inliers = list(filter(lambda x: x < self.threshold, residuals))

            inliers_acheived_so_far.append(len(inliers))
            colors.blue(len(inliers))

            if len(inliers) > best["inliers"]:
                best["inliers"] = len(inliers)
                best["indexes"] = indexes
                print(len(inliers))

            new_pheromone_row = []
            for i, point in enumerate(points):
                delta = self.pheromone_delta(points, len(inliers), np.mean(inliers_acheived_so_far),
                                             len(inliers_acheived_so_far) + 1, residuals[i], self.threshold)
                delta = delta
                new_pheromone_row.append((self.evaporation * pheromone_matrix[-1][i]) + delta)
            pheromone_matrix.append(new_pheromone_row)
        return best

    def transform(self, couples, people=None):
        if people == None:
            people = data.get.people_raw()
        X, y = data.make.couples_xy(couples, people)
        points = list(zip(X, y))
        best = self.ANTSAC(points)

    def MAD(self, arr):
        arr = np.ma.array(arr).compressed()
        med = np.median(arr)
        return np.median(np.abs(arr - med))

    def calc_resids(self, actual, predicted):
        distances = euclidean_distances(actual, predicted)
        distances = [distance[0] for distance in distances]
        return distances


antsac = ANTSAC(max_iter=500)
couples = data.get.couples_raw()
couples = preprocessing.couples_raw.Mirror().transform(couples)
print(antsac.transform(couples))
