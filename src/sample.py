import cv2 as cv

class Sample:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.feature_vector = []
        self.feature_vector_type = ''

    def extract_sift(self):
        ver = cv.getVersionString()
        sift = cv.xfeatures2d_SIFT.create()
        kp, des = sift.detectAndCompute(self.data, None)

        return kp
