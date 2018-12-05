# Import necessary packages
import scipy
import os
import math
import numpy as np

import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def load_data_filenames(data_dir):
    label_counter = 0
    training_images = []
    training_labels = []
    for subdir, dirs, files in os.walk(data_dir):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    training_images.append(os.path.join(folder_subdir, file))
                    training_labels.append(label_counter)

            label_counter = label_counter + 1
    return training_images, training_labels

def load_imagenet(batch_size=128, num_classes=1000, new_shape=(32,32,3), data_dir="./datasets/ImageNet/train"):
    training_images, training_labels = load_data_filenames(data_dir)
    nice_n = math.floor(len(training_images) / batch_size) * batch_size
    B = np.zeros(shape=(nice_n,)+new_shape )
    L = np.zeros(shape=(nice_n))

    for index in range(nice_n):
        img = load_img(training_images[index])
        new_img = img_to_array(img)
        new_img = scipy.misc.imresize(new_img, new_shape)
        B[index] =  new_img
        B[index] /= 255

        L[index] = training_labels[index]

    return B, keras.utils.to_categorical(L, num_classes)