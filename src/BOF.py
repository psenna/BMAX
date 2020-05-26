import cv2 as cv
import os
import numpy as np
from .sample import Sample


class BOF:
    def __init__(self, images, labels):
        if len(images) != len(labels):
            self.valid = False
            return

        self.samples = []
        for i in range(len(images)):
            sample = Sample(images[i], labels[i])
            sample.extract_sift()
            self.samples.append(sample)

        vocabulary_size = os.getenv('VOCABULARY_SIZE')
        if vocabulary_size is None:
            vocabulary_size = '200'
        self.vocabulary_size = int(vocabulary_size)

        training_size_per_label = os.getenv('TRAINING_SIZE_PER_LABEL')
        if training_size_per_label is None:
            training_size_per_label = '100'
        self.training_size = int(training_size_per_label)

        test_max_size = os.getenv('TEST_MAX_SIZE')
        if test_max_size is None:
            test_max_size = '999999999999'
        self.test_max_size = int(test_max_size)

        self.vocabulary = []
        self.valid = True

    def select_training_test_data(self):
        #todo Separa por classes

        #todo Permutacao aleatória por classe

        #todo split pela quantidade de amostras de treino por classe

        #todo unir os vetores de trino e teste
        self.training = []
        self.test = []

    def create_vocabulary(self, descriptors):
        #todo create the vocabulary with training data
        return
