import unittest


class SampleTest(unittest.TestCase):

    def test_find_dense_keypoints(self):
        import cv2 as cv
        import os
        from src.sample import Sample

        os.environ.clear()
        os.environ['KEYPOINT_DETECTOR_TYPE'] = "DENSE"

        img = cv.imread('../data/gopher.jpg', cv.IMREAD_GRAYSCALE)
        sample = Sample(img, 1)
        sample.find_keypoits()
        self.assertEqual(len(sample.keypoints), 6942)

    def test_find_dense_keypoints_with_step(self):
        import cv2 as cv
        import os
        from src.sample import Sample

        os.environ.clear()
        os.environ['KEYPOINT_DETECTOR_TYPE'] = "DENSE"
        os.environ['DENSE_DETECTOR_STEP'] = "50"

        img = cv.imread('../data/gopher.jpg', cv.IMREAD_GRAYSCALE)
        sample = Sample(img, 1)
        sample.find_keypoits()
        self.assertEqual(len(sample.keypoints), 288)

    def test_find_sift_keypoints(self):
        import cv2 as cv
        import os
        from src.sample import Sample

        os.environ.clear()
        os.environ['KEYPOINT_DETECTOR_TYPE'] = "SIFT"

        img = cv.imread('../data/gopher.jpg', cv.IMREAD_GRAYSCALE)
        sample = Sample(img, 1)
        sample.find_keypoits()
        self.assertEqual(len(sample.keypoints), 1613)

    def test_extraxt_sift_descriptor(self):
        import cv2 as cv
        import os
        from src.sample import Sample

        os.environ.clear()

        img = cv.imread('../data/gopher.jpg', cv.IMREAD_GRAYSCALE)
        sample = Sample(img, 1)
        sample.extract_sift()
        self.assertEqual(len(sample.keypoints), 1613)
        self.assertEqual(sample.descriptors[1].shape, (1613, 128))

    def test_extraxt_sift_descriptor_with_dense_detector(self):
        import cv2 as cv
        import os
        from src.sample import Sample

        os.environ.clear()
        os.environ['KEYPOINT_DETECTOR_TYPE'] = "DENSE"

        img = cv.imread('../data/gopher.jpg', cv.IMREAD_GRAYSCALE)
        sample = Sample(img, 1)
        sample.extract_sift()
        self.assertEqual(len(sample.keypoints), 6942)
        self.assertEqual(sample.descriptors[1].shape, (6942, 128))

    def test_create_feature_vector_BOF(self):
        import cv2 as cv
        import os
        from src.sample import Sample
        from src.sample_utils import SampleUtils

        os.environ.clear()
        os.environ['KEYPOINT_DETECTOR_TYPE'] = "DENSE"

        img = cv.imread('../data/gopher.jpg', cv.IMREAD_GRAYSCALE)
        sample = Sample(img, 1)
        vocabulary = SampleUtils.create_vocabulary([sample], 10)

        self.assertEqual(vocabulary.shape, (10, 128))

        feature_vector = sample.get_feature_vector(vocabulary)

        self.assertEqual(len(feature_vector), 10)
