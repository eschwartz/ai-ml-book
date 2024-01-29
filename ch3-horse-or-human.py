# Horse or human data set
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/horses_or_humans.py
#
# Run in Google Colab:
#   https://bit.ly/horsehuman
import zipfile
from urllib import request

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import RMSprop

training_dir = 'data/horse-or-human/training'
validation_dir = "data/horse-or-human/validation"

# Create a dataset from the images
train_datagen = ImageDataGenerator(
    # All images will be rescaled by 1/255
    rescale=1 / 255,
    # Image augmentation
    # make some random changes to the training images
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),  # images are 300x300 pixels
    class_mode='binary',  # two types of images (vs. "categorical" with many types)
)

# Create validation dataset
validation_datagen = ImageDataGenerator(rescale=1 / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),  # images are 300x300 pixels
    class_mode='binary',  # two types of images (vs. "categorical" with many types)
)

# Create the neural network
model = keras.models.Sequential([
    # Convolutions (~photoshop filters)
    # See ch3-img-feats.py
    # note that input shape now has 3 channels (RGB)
    Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),

    # Pooling (~downsizing)
    MaxPooling2D(2, 2),

    # More rounds of convolutions/pooling
    # use many layers because we want to drive down to many smaller images,
    # each with features highlighted
    # Results in 7x7 images, which are hopefully "activated feature maps"
    # with 1.7 million parameters
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Flatten to a 1d set of values
    Flatten(),

    # Run through 512 nodes
    Dense(512, activation='relu'),

    # Output to a single neuron, for the binary result
    # sigmoid activation function works to drive values to 0 or 1
    Dense(1, activation='sigmoid'),
])

# this is tweaked a bit, to work better for binary outputs
model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),  # lr is learning rate
    metrics=['accuracy']
)

# fit the model to the training data
# using the file-based data generator we setup earlier
history = model.fit(
    train_generator,
    epochs=15,
    # steps_per_epoch=8,
    verbose=1,
    validation_data=validation_generator,
)

model.summary(line_length=80)
