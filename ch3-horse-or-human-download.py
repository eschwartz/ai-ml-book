# Horse or human data set
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/horses_or_humans.py
import zipfile
from urllib import request

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import RMSprop

# Download training data
# url = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
#
# file_name = "horse-or-human.py"
# training_dir = 'data/horse-or-human/training'
#
# request.urlretrieve(url, file_name)
# zip_ref = zipfile.ZipFile(file_name, 'r')
# zip_ref.extractall(training_dir)
# zip_ref.close()

# Download validation data
validation_url = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
validation_dir = "data/horse-or-human/validation/"
validation_file_name = "validation-horse-or-human.zip"
request.urlretrieve(validation_url, validation_file_name)
zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()
