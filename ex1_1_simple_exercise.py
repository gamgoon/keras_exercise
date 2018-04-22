import keras
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y = x * 2 + 1

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile('SGD', 'mse')

model.fit(x[:1], y[:1], epochs=1000, verbose=0)

print('Targets:', y[2:])
print('Predictions:', model.predict(x[2:]).flatten())

