import numpy as np

# keras cannot be install when there's matplotlib installed! Same for the opposite situation!
# You should execute this file in a different environment (mlEnv or nnxor)!
# UPDATE!!! It is now completely possible to install matplotlib and keras in one place!
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD


np.random.seed(444)

x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])


model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(learning_rate=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x, y, batch_size=1, epochs=5000)

if __name__ == '__main__':
    print(model.predict(x))
