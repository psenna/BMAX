import cv2 as cv
import os

class Sample:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.feature_vector = []
        self.feature_vector_type = ''
        self.keypoints = []
        self.descriptors = []

    def find_keypoits(self):
        keypoin_detector_type = os.getenv('KEYPOINT_DETECTOR_TYPE')

        if (keypoin_detector_type == None):
            keypoin_detector_type = "SIFT"

        if (keypoin_detector_type == "DENSE"):
            dense_detector_step = os.getenv("DENSE_DETECTOR_STEP")
            if(dense_detector_step == None):
                dense_detector_step = 10
            if(isinstance(dense_detector_step, str)):
                dense_detector_step = int(dense_detector_step)

            self.keypoints = [cv.KeyPoint(x, y, dense_detector_step) for y in range(0, self.data.shape[0], dense_detector_step)
                  for x in range(0, self.data.shape[1], dense_detector_step)]
            return

        sift_detector = cv.xfeatures2d_SIFT.create()
        self.keypoints = sift_detector.detect(self.data)
        return



    def extract_sift(self):

        if(len(self.keypoints) == 0):
            self.find_keypoits()

        sift = cv.xfeatures2d_SIFT.create()
        self.descriptors = sift.compute(self.data, self.keypoints)

        return
