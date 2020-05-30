import os
from .sample import Sample
from .sample_utils import SampleUtils
from .svm import Svm


class BOF:
    def __init__(self, images, labels):
        if len(images) != len(labels):
            self.valid = False
            return

        self.samples = []
        for i in range(images.shape[0]):
            sample = Sample(images[i], labels[i])
            sample.get_descriptors()
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

    def run(self):
        # Split dataset
        training_samples, test_samples = SampleUtils.split_training_test_data(self.samples, self.training_size, self.test_max_size)

        print("Creating vocabulary")
        self.vocabulary = SampleUtils.create_vocabulary(training_samples, self.vocabulary_size)
        print("Vocabulary created")

        print("Extracting features")
        training_feature_vector, training_labels = SampleUtils.get_samples_feature_vector_and_labels(training_samples, self.vocabulary)
        print("Feature vectors created")

        print("Training SVM")

        self.svm = Svm()

        self.svm.grid_search(training_feature_vector, training_labels)

        self.svm.train(training_feature_vector, training_labels)

        print("SVM Trained")

        print("Testing SVM")

        test_feature_vector, test_labels = SampleUtils.get_samples_feature_vector_and_labels(test_samples, self.vocabulary)

        self.svm.test(test_feature_vector, test_labels)

        test_prediction = self.svm.predict(test_feature_vector)

        test_err = (test_prediction.astype(int) != test_labels).mean()
        print('Testing Error: %.2f %%' % (test_err * 100))
