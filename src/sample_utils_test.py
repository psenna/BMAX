import unittest


class SampleUtilsTest(unittest.TestCase):

    def test_split_samples_by_label(self):
        import numpy as np
        import os
        from src.sample import Sample
        from src.sample_utils import SampleUtils

        os.environ.clear()

        samples = []
        for i in range(12):
            samples.append(Sample(np.random.rand(50, 50), str(i % 3)))

        label_separated_samples = SampleUtils.split_samples_by_label(samples)

        for i in range(3):
            test_samples = label_separated_samples[str(i)]
            self.assertEqual(4, len(test_samples))
            for j in range(len(test_samples)):
                self.assertEqual(str(i), test_samples[j].label)

    def test_split_training_test_data(self):
        import numpy as np
        import os
        from src.sample import Sample
        from src.sample_utils import SampleUtils

        os.environ.clear()

        samples = []
        for i in range(12):
            samples.append(Sample(np.random.rand(50, 50), str(i % 3)))

        training_data, test_data = SampleUtils.split_training_test_data(samples, 1, 3)

        self.assertEqual(3, len(training_data))

        self.assertEqual(9, len(test_data))

        for i in range(len(test_data)):
            self.assertTrue(test_data[i] not in training_data)

        self.assertFalse(samples == training_data+test_data)

    def test_split_training_test_data_with_limited_test(self):
        import numpy as np
        import os
        from src.sample import Sample
        from src.sample_utils import SampleUtils

        os.environ.clear()

        samples = []
        for i in range(12):
            samples.append(Sample(np.random.rand(50, 50), str(i % 3)))

        training_data, test_data = SampleUtils.split_training_test_data(samples, 1, 2)

        self.assertEqual(3, len(training_data))

        self.assertEqual(6, len(test_data))

    def test_create_vocabulary(self):
        import cv2 as cv
        import os
        from src.sample import Sample
        from src.sample_utils import SampleUtils

        os.environ.clear()
        os.environ['KEYPOINT_DETECTOR_TYPE'] = "DENSE"

        img = cv.imread('../data/gopher.jpg', cv.IMREAD_GRAYSCALE)
        sample = Sample(img, 1)
        sample2 = Sample(img, 1)
        vocabulary = SampleUtils.create_vocabulary([sample, sample2], 10)

        self.assertEqual(vocabulary.shape, (10, 128))
