import data, math
import numpy as np

X, names = data.get.people_xy()

from sklearn.cluster import AgglomerativeClustering as Clusterer
clusterer = Clusterer(n_clusters=16)
clusterer = clusterer.fit(X)
print(clusterer.labels_)

from sklearn.decomposition import PCA, ProjectedGradientNMF, TruncatedSVD
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS, TSNE
X_transformed = SpectralEmbedding(n_components=2).fit_transform(X)


#okay so we need to make a grid
side_len = math.ceil(math.sqrt(len(X_transformed)))
from sklearn.preprocessing import MinMaxScaler
grid = []
from sklearn.metrics.pairwise import euclidean_distances
center = [side_len/2] * 2
for x in range(side_len):
    for y in range(side_len):
        d = euclidean_distances([[x,y]], [center])[0][0]
        grid.append([[x,y], d])
grid.sort(key=lambda x:x[1])
grid = grid[0:len(X_transformed)]
print(len(grid))
print(len(X_transformed))
#now we trim the corners so len(grid) == len(points)
#we need to sort the grid by distance to center
scaler = MinMaxScaler(feature_range=(0, side_len))
X_transformed = scaler.fit_transform(X_transformed)

#now we compute a cost matrix
cost_matrix = []
for person in X_transformed:
    costs = []
    for grid_point in grid:
        costs.append(euclidean_distances([person], [grid_point[0]])[0][0])
    cost_matrix.append(costs)

#now we use linear assignment to solve this matrix
from sklearn.utils.linear_assignment_ import linear_assignment
solutions = linear_assignment(np.array(cost_matrix))
print(solutions)
new_X = [[]] * len(X_transformed)
for solution in solutions:
    new_X[solution[0]] = grid[solution[1]][0]
X_transformed = new_X
print(X_transformed)

plt_x = []
plt_y = []
plt_z = list(map(lambda x:x+1, clusterer.labels_))
for a in X_transformed:
    plt_x.append(a[0])
    plt_y.append(a[1])

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for i, txt in enumerate(names):
    txt = txt.split(" ")[0]
    ax.annotate(txt, (X_transformed[i][0] + .07, X_transformed[i][1] - .1))

ax.scatter(plt_x, plt_y, c=plt_z, s=40, cmap="nipy_spectral")

couples = data.get.couples_raw()
for couple in couples:
    x_index = names.index(couple["male"])
    y_index = names.index(couple["female"])
    a = X_transformed[x_index]
    b = X_transformed[y_index]
    line = plt.plot([a[0], b[0]], [a[1], b[1]], c="grey")
    plt.setp(line, linewidth=.5)

plt.show()
