# Models may overfit
# certain neurons may be overfit to the training data,
# which will have a disparate impact on the overall model
#
# Using dropout layers will randomly remove certain neurons from a layer
# and prevent overfitting
import tensorflow as tf

import keras

# Load MNIST fashion dataset
data = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# "normalizing"
training_images = training_images / 255
test_images = test_images / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(
    # More performant version of stochastic gradient descent (sdg)
    optimizer='adam',
    # How to calculate error between predicted and actual value
    # This one is good for choosing categories
    loss='sparse_categorical_crossentropy',
    # print out the accuracy while it's training
    metrics=['accuracy'],
)

# train the network
model.fit(training_images, training_labels, epochs=10)

# evaluate the model against our test data
print("\nevaluating against test data")
# Will print out the accuracy of our model against the test data
model.evaluate(test_images, test_labels)

# Classify each test image using our model
print("\nclassifying model against test images")
classifications = model.predict(test_images)
print("first image classifications:")
print(classifications[0])
print(f"actual label: {test_labels[0]}")

"""
Without dropouts:
Epoch 10/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2317 - accuracy: 0.9132

evaluating against test data
313/313 [==============================] - 1s 1ms/step - loss: 0.3411 - accuracy: 0.8808

^ The accuracy of the model against the training data is higher than 
against the test data. That means we're over-fit


With dropouts:
Epoch 10/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.3065 - accuracy: 0.8886

evaluating against test data
313/313 [==============================] - 1s 2ms/step - loss: 0.3386 - accuracy: 0.8815


Note that the training data accuracy is now lower,
but it's closer to the test data accuracy. 
"""
