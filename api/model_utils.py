from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.optimizers import Adam


def bl_nn(x_matrix):
    optimizer = Adam(0.0001)
    model = Sequential()
    model.add(
        Dense(
            x_matrix.shape[1] * 20,
            input_dim=x_matrix.shape[1],
            kernel_initializer='normal')
    )
    model.add(LeakyReLU())
    model.add(
        Dense(
            x_matrix.shape[1] * 20,
            kernel_initializer='normal')
    )
    model.add(LeakyReLU())
    model.add(
        Dense(x_matrix.shape[1] * 20,
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
