import pandas
from pandas.tools.plotting import scatter_matrix
import data
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AffinityPropagation as Clusterer
clusterer = Clusterer()


X, y = data.get.people_xy()
vecs = []
names = []
couples = data.get.couples_raw()
for couple in couples:
    vecs.append(np.array(X[y.index(couple["male"])]) - np.array(X[y.index(couple["female"])]))
    names.append(couple["male"].split(' ')[0] + " - " + couple["female"].split(" ")[0])

labels = Clusterer().fit(X).labels_

df = pandas.DataFrame(vecs, columns=["Extroversion", "Emotional", "Agreeableness","Conscientiousness","Intellect"])

plot = scatter_matrix(df,figsize=(15,15),marker='o',hist_kwds={'bins':10},s=60,alpha=1,cmap="nipy_spectral")

plt.show()