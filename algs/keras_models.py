from wrappers import KerasWrapper

def gen_keras():
    from keras.layers import Dense, Dropout

    models = []

    models.append([
        Dense(5, input_shape=(5,)),
        Dense(5),
        Dense(5),
    ])

    models.append([
        Dense(5, input_shape=(5,)),
        Dense(5),
        Dropout(.5),
        Dense(5)
    ])

    for model in models:
        for s in [False]:
            yield KerasWrapper(model, scale_importance=s)