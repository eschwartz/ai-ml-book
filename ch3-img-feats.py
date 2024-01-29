# This builds on the image recognition in ch2
# adding some feature detection
# This is basically "messing around with photoshop filters" to make important
# features in the image more recognizable
#
# For example, the model may decide to sharpen each image by 20% and lighten by 5%
# before predicting its label
import tensorflow as tf

import keras

# Load MNIST fashion dataset
# Provides 60,000 training images and 1,000 test images
# Returns arrays of 28x28 pixels values, on a 0-255 greyscale
# Labels is an array of 60,000 0-9 values, representing the image label
data = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# Reshape image data
# Convolution will expect a 3rd dimension to the images (color)
# reshaping adds this dimension
training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# "normalizing"
# Convert each pixel value to a float between 0 and 1
# by dividing each value in the array by 255
# improves performance and accuracy because math
training_images = training_images / 255
test_images = test_images / 255

# Setup the neural network
model = keras.Sequential([
    # Add a convolution layer
    # A "convolution" is like a Photoshop filter
    # it modifies each pixel based on the values of surrounding pixels
    # NN will keep trying different filter values, to better match the training images
    keras.layers.Conv2D(
        64,  # number of convultions to apply to the image
        (3, 3),  # size of the filter (how many surrounding pixels to look at)
        # note that a 3x3 filter will remove a 1 pixel border around the image, making it 26x26
        activation='relu',
        input_shape=(28, 28, 1),  # 28x28 image, 1 for greyscale
    ),

    # Add a pooling layer
    # "pooling" downsizes the image, where each new pixel value
    # will be a function of the surrounding pixel values
    # "Max" pooling means take the max value (lightest pixel) from a group
    keras.layers.MaxPooling2D(2, 2),  # 2x2 area of pixels will be downsized to a single pixel

    # Repeat the convolution and pooling layers
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    # same as Ch2
    keras.layers.Flatten(),  # flatten 2x2 pixels to a list of pixel values
    keras.layers.Dense(128, activation=tf.nn.relu),  # run through 128 neurons (parameters)
    keras.layers.Dense(10, activation=tf.nn.softmax),  # output 10 possible lable values
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# train the network
model.fit(training_images, training_labels, epochs=3)  # 50 epochs will ge ~99% accuracy now, but is slow to train

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

# this will show how image size is reduced when filters + pooling are applied
# End result is many 5x5 images (~37,000, see "Param #" column)
model.summary()

"""
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
# 64 convolutions.
# for each one, we need to set a value for each filter in a 3x3 grid
# So that's 9 parameters, plus a bias (mX + c)
# So 10 params x 64 convolutions = 640 parameters to learn
conv2d (Conv2D)             (None, 26, 26, 64)        640

# Pooling just reduces the image size, no more parameters to learn
max_pooling2d (MaxPooling2D)  (None, 13, 13, 64)        0

# 64 more convolution filters
# applied to each of the previous filters
# So (64 x (64 x 9)) + 64 = 36,928
conv2d_1 (Conv2D)           (None, 11, 11, 64)        36928

# 64 5x5 images
max_pooling2d_1 (MaxPooling2D)  (None, 5, 5, 64)          0

Flatten those 64 5x5 images, =1,600 pixel values
flatten (Flatten)           (None, 1600)              0

# We have a weight and bias for each pixel value
# so 1,600 x 128 + 128 = 204,928
dense (Dense)               (None, 128)               204928

# 128 x 10 + 128 = 1,290
dense_1 (Dense)             (None, 10)                1290

=================================================================

# Training the network means getting the best possible value for each of these
# 243,786 parameters
Total params: 243786 (952.29 KB)
Trainable params: 243786 (952.29 KB)
Non-trainable params: 0 (0.00 Byte)
"""
