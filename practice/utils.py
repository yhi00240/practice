import numpy as np

def image_to_mnist(image_data):
    return ((255 - np.array(image_data, dtype=np.int)) / 255.0).reshape(1, 784)

def mnist_to_image(mnist_data):
    return 255 - np.array(mnist_data * 255, dtype=np.int)
