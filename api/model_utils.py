from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.optimizers import Adam


def bl_nn():
    optimizer = Adam(0.0001)
    model = Sequential()
    model.add(
        Dense(
            660,
            input_dim=33,
            kernel_initializer='normal')
    )
    model.add(LeakyReLU())
    model.add(
        Dense(
            660,
            kernel_initializer='normal')
    )
    model.add(LeakyReLU())
    model.add(
        Dense(660,
              kernel_initializer='normal')
    )
    model.add(LeakyReLU())
    model.add(
        Dense(1, kernel_initializer='normal')
    )
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer
    )
    return model
