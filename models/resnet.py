# Import necessary packages
import keras
import keras_resnet.models
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

def resnet_model (img_shape=(224, 224, 3), n_classes=1000):
    return keras_resnet.models.ResNet50(img_shape, classes=n_classes)




