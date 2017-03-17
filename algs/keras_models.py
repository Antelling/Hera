from wrappers import KerasWrapper

def gen_keras():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    models = []

    models.append([])
    for _ in range(0, 2):
        model = Sequential()
        model.add(Dense(5, input_shape=(5,)))
        model.add(Dense(5))
        model.add(Dense(5))
        model.compile(loss="mse", optimizer="adam")
        models[-1].append(model)

    models.append([])
    for _ in range(0, 2):
        model = Sequential()
        model.add(Dense(5, input_shape=(5,)))
        model.add(Dense(5))
        model.add(Dropout(.4))
        model.add(Dense(5))
        model.compile(loss="mse", optimizer="adam")
        models[-1].append(model)

    for model in models:
        yield KerasWrapper(model)