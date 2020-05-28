import cv2 as cv
import numpy as np
import os


class Sample:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.feature_vector = None
        self.keypoints = None
        self.descriptors = None

    # SIFT is the default keypoint detector
    def find_keypoits(self):
        keypoint_detector_type = os.getenv('KEYPOINT_DETECTOR_TYPE')

        if keypoint_detector_type is None:
            keypoint_detector_type = "SIFT"

        if keypoint_detector_type == "DENSE":
            dense_detector_step = os.getenv("DENSE_DETECTOR_STEP")
            if dense_detector_step is None:
                dense_detector_step = 10
            if isinstance(dense_detector_step, str):
                dense_detector_step = int(dense_detector_step)

            self.keypoints = [cv.KeyPoint(x, y, dense_detector_step) for y in
                              range(0, self.data.shape[0], dense_detector_step)
                              for x in range(0, self.data.shape[1], dense_detector_step)]
            return

        sift_detector = cv.xfeatures2d_SIFT.create()
        self.keypoints = sift_detector.detect(self.data)
        return

    def extract_sift(self):

        if self.keypoints is None:
            self.find_keypoits()

        sift = cv.xfeatures2d_SIFT.create()
        self.descriptors = sift.compute(self.data, self.keypoints)
        return

    # SIFT is the default feature extractor
    def extract_descriptors(self):
        descriptor_type = os.getenv("DESCRIPTOR_TYPE")
        if descriptor_type is None:
            descriptor_type = "SIFT"

        # Other extractors

        self.extract_sift()


    def get_descriptors(self):
        if self.descriptors is None:
            self.extract_descriptors()

        return self.descriptors[1]

    def get_feature_vector(self, vocabulary: []):
        if (vocabulary is None or len(vocabulary) == 0) and self.feature_vector is not None:
            return self.feature_vector

        feature_type = os.getenv('FEATURE_TYPE')
        if feature_type is None:
            feature_type = 'BOF'

        if feature_type == 'BOF':
            if (vocabulary is None or len(vocabulary) == 0) and self.feature_vector is None:
                print("BOF need a vocabulary to be computed")
                return None
            self.get_bof_feature_vector(vocabulary)

        return self.feature_vector

    def get_bof_feature_vector(self, vocabulary: []):
        self.feature_vector = np.zeros(vocabulary.shape[0])

        distances = self.distance(self.get_descriptors(), vocabulary)

        min_index = np.argmin(distances, axis=1)

        for i in range(len(min_index)):
            self.feature_vector[min_index[i]] += 1

        self.feature_vector /= sum(self.feature_vector)
        return

    @classmethod
    def distance(cls, features, vocabulary):
        return np.array([[np.linalg.norm(i - j) for j in vocabulary] for i in features])