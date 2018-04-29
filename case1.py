from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import np_utils

model = Sequential()
model.add(Dense(16, input_dim=1, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
Y = np.array([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]])
model.fit(X, Y, epochs=200, batch_size=10)

X_hat = np.array([[2]])
Y_hat = model.predict(X_hat)
print(Y_hat)