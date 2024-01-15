# image recognition, using the mnist fashion dataset
# https://www.tensorflow.org/datasets/catalog/fashion_mnist
# Each neuron calculates a Wx+B "parameter"
# or mx+c
# which are summed together to create the model
import tensorflow as tf

import keras

# Load MNIST fashion dataset
# Provides 60,000 training images and 1,000 test images
# Returns arrays of 28x28 pixels values, on a 0-255 greyscale
# Labels is an array of 60,000 0-9 values, representing the image label
data = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# "normalizing"
# Convert each pixel value to a float between 0 and 1
# by dividing each value in the array by 255
# improves performance and accuracy because math
training_images = training_images / 255
test_images = test_images / 255

model = keras.Sequential([
    # Note actually a layer of neurons, just an input specification
    # Input images are 28x28 pixels
    # Flatten takes 28x28 "square" data, and flattens it to a line of 784 vals
    keras.layers.Flatten(input_shape=(28, 28)),
    # 128 neurons
    # each neuron has internal params randomly initialized
    # number of neurons is somewhat arbitrary
    # too few leads to bad fit and slow run, too many to overfitting
    # see "hyperparameter tuning"
    #
    # activation fn runs on each neuron
    # "relu" (rectified linear unit) enforces that values are >0
    keras.layers.Dense(128, activation=tf.nn.relu),
    # Output layer
    # Has 10 units, bc we have 10 possible output values (different item types)
    # softmax picks the prev neuron with the highest probability of matching the item type
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


# Setup a "callback" so that we can stop training
# when the accuracy gets to a certain level
# (as opposed to hard-coding the number of epochs)
class ModelCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.95:
            print("\n Reached 95% accuracy so cancelling training!")
            self.model.stop_training = True


callback = ModelCallback()

# train the network
model.fit(training_images, training_labels, epochs=50, callbacks=[callback])

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
# output
# [3.6966164e-06 4.6526722e-07 7.2630619e-06 6.9922626e-06 8.2189848e-07
#  2.1274818e-02 1.7656952e-04 7.0512958e-02 5.1711377e-05 9.0796471e-01]
# actual label: 9
#
# These are the values of each output node
# Note that node #9 has a value closest to 1 (rest are near 0)
