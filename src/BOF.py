import cv2 as cv
import numpy as np
from .sample import Sample

class BOF:
    def __init__(self, images, labels, vocabulary_size, training_size_per_label, test_max_size):
        if(len(images) != len(labels)):
            return  False
        self.samples = []
        for i in range(len(images)):
            self.samples.append(Sample(images[i], labels[i]))
        self.vocabulary_size = vocabulary_size
        self.training_size = training_size_per_label
        self.test_max_size = test_max_size
        return True

    def select_training_test_data(self):
        #todo Separa por classes

        #todo Permutacao aleatória por classe

        #todo split pela quantidade de amostras de treino por classe

        #todo unir os vetores de trino e teste
        self.training = []
        self.test = []

    def create_vocabulary(self):
        #todo create the vocabulary with training data
        return



# Test
