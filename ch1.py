import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense

# First layer in neural network
# with a single neuron
l0 = Dense(units=1, input_shape=[1])

model = Sequential([l0])
# loss = how to calculate error of values using the predicted formula (model) vs. actual data
# optimizer = given a certain loss, how to modify the model (formula)
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))

# Weights will show model as
# weight * x + bias  (WX+B)
# Should be close to 2x - 1
print(f"{l0.get_weights()}")
