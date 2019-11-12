from tensorflow import keras
from datatools.tfrecord import convert_to_tfrecord
import cv2 as opencv
import numpy as np


def get_cifar10_dataset(width, height):
    (x_train, y_train), (x_valid, y_valid) = keras.datasets.cifar10.load_data()

    x_train = np.array([opencv.resize(img, (width, height)) for img in x_train[:, :, :, :]])
    x_valid = np.array([opencv.resize(img, (width, height)) for img in x_valid[:, :, :, :]])

    print(y_train.shape)

    print("Changing types")
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')

    print("Scaling")
    x_train /= 255.0
    x_valid /= 255.0

    return x_train, y_train, x_valid, y_valid


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = get_cifar10_dataset(224, 224)

    convert_to_tfrecord(x_valid, y_valid, "output/cifar10.tfrecord")