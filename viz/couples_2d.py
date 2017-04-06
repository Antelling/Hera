import data, math
import numpy as np

people = data.get.people_raw()
couples = data.get.couples_raw()
new_X = []
new_names = []
for couple in couples:
    m = np.array(people[couple["male"]]["position"])
    f = np.array(people[couple["female"]]["position"])
    vec = m - f
    new_X.append(vec)
    new_names.append(couple["male"].split(" ")[0] + "/" + couple["female"].split(' ')[0])
X = new_X
names = new_names

from sklearn.cluster import AgglomerativeClustering as Clusterer
clusterer = Clusterer(n_clusters=16)
clusterer = clusterer.fit(X)
print(clusterer.labels_)

from sklearn.decomposition import PCA, ProjectedGradientNMF, TruncatedSVD
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS, TSNE
X_transformed = MDS(n_components=2).fit_transform(X)

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
    ax.annotate(txt, (X_transformed[i][0], X_transformed[i][1]))

ax.scatter(plt_x, plt_y, c=plt_z, s=40, cmap="nipy_spectral")

plt.show()
