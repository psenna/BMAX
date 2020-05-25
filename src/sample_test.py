import unittest

class SampleTest(unittest.TestCase):

    def test_extrair_descritor_sift(self):
        import cv2 as cv
        from src.sample import Sample
        ver = cv.getVersionMajor()
        img = cv.imread('../data/gopher.jpg', cv.IMREAD_GRAYSCALE)
        sample = Sample(img, 1)
        kp = sample.extract_sift()
        self.assertEqual(len(kp), 1613)


if __name__ == '__main__':
    unittest.main()
