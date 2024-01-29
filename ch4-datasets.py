import tensorflow as tf
import tensorflow_datasets as tfds

mnist_data = tfds.load("fashion_mnist")

mnist_train, info = tfds.load(name="fashion_mnist", split="train", with_info=True)

for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())
    print(item['image'])
    print(item['label'])

print('done')
