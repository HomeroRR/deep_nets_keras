import keras

def load_mnist(num_classes=1000):
    (X, Y), (_, _) = keras.datasets.mnist.load_data()
    return X, keras.utils.to_categorical(Y, num_classes)