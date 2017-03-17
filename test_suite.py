import wrappers
import data
from keras.models import Sequential
from keras.layers import Dense

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
predictor = wrappers.SklearnWrapper(model)

couples = data.get.couples_xy()
predictor.fit(*couples)
print(predictor.predict([{"position": [8, 8, 8, 8, 8], "gender": "male"}]))
print(predictor.predict([{"position": [8, 8, 8, 8, 8], "gender": "female"}]))

models = []
for _ in range(0, 2):
    model = Sequential()
    model.add(Dense(5, input_shape=(5,)))
    model.add(Dense(5))
    model.add(Dense(5))
    model.compile(loss="mse", optimizer="adam")
    models.append(model)

predictor = wrappers.KerasWrapper([models[0], models[1]], epochs=2000)
predictor.fit(*couples)
print(predictor.predict([{"position": [8, 8, 8, 8, 8], "gender": "male"}]))
print(predictor.predict([{"position": [8, 8, 8, 8, 8], "gender": "female"}]))
