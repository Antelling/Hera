import data, preprocessing
people = data.get.people_raw()
#people = preprocessing.people.Flatten().transform(people)
X, y = data.make.people_xy(people)

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

df = pandas.DataFrame(X, columns=["Extroversion", "Emotional", "Agreeableness","Conscientiousness","Intellect"])

plot = scatter_matrix(df,figsize=(15,15),marker='o',hist_kwds={'bins':10},s=60,alpha=1,cmap="nipy_spectral")

plt.show()
