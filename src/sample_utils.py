import numpy as np
import cv2 as cv
import os
from itertools import islice
from .sample import Sample


class SampleUtils:
    @classmethod
    def split_samples_by_label(cls, samples: [Sample]):
        label_dic = {}

        for i in range(len(samples)):
            if samples[i].label not in label_dic:
                label_dic[samples[i].label] = []

            label_dic[samples[i].label].append(samples[i])

        return label_dic

    @classmethod
    def split_training_test_data(cls, samples, training_per_label, max_test_per_label):
        training_data = []
        test_data = []

        samples_by_label = SampleUtils.split_samples_by_label(samples)

        for key in samples_by_label:
            samples_by_label[key] = list(np.random.permutation(samples_by_label[key]))
            training_data += list(islice(samples_by_label[key], training_per_label))
            limit = min(len(samples_by_label[key])-training_per_label, max_test_per_label)
            test_data += list(islice(samples_by_label[key], training_per_label, training_per_label + limit))

        return training_data, test_data

    # Create the dictionary with kmeans
    @classmethod
    def create_vocabulary(cls, samples: [Sample], vocabulary_size: int):
        descriptors = None
        for i in range(len(samples)):
            var = samples[i].get_descriptors()
            if descriptors is None:
                descriptors = samples[i].get_descriptors()
            else:
                descriptors = np.concatenate((descriptors, samples[i].get_descriptors()), axis=0)

        kmeans_max_iter = os.getenv('KMEANS_MAX_ITER')
        if kmeans_max_iter is None:
            kmeans_max_iter = 10

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, kmeans_max_iter, 1.0)
        ret, label, center = cv.kmeans(descriptors, vocabulary_size, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        return center
