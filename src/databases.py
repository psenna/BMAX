import numpy as np
import pickle


class Databases:

    @classmethod
    def get_minist(cls0, path):
        file = open(path + 'databases/mnist.pkl', 'rb')

        ((x_train, y_train), (x_test, y_test), _) = pickle.load(file, encoding='latin-1')

        file.close()

        images = np.concatenate((x_train * 255, x_test * 255)).reshape((-1, 28, 28)).astype(np.uint8)
        labels = np.concatenate((y_train, y_test))

        return images, labels
