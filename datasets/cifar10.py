import keras

def load_cifar10(num_classes=1000):
    (X, Y), (_, _) = keras.datasets.cifar10.load_data()
    return X, keras.utils.to_categorical(Y, num_classes)