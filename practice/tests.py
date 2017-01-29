from django.test import TestCase

from practice.services import MNIST


class MNISTTestCase(TestCase):

    def testWholeOperation(self):
        mnist = MNIST()

        mnist.loadTrainingData()
        assert mnist.training_data is not None

        mnist.setAlgorithm()
        assert mnist.hypothesis is not None
        assert mnist.inference is not None
        assert mnist.cost is not None

        mnist.setTraining()
        assert mnist.learning_rate is not None
        assert mnist.optimizer is not None
        assert mnist.training_epochs is not None
        assert mnist.batch_size is not None
        assert mnist.train_operation is not None
        assert mnist.accuracy_operation is not None
        assert mnist.summary_operation is not None

        mnist.run()

        mnist.loadTestingData()
        assert mnist.test_data is not None

        mnist.test()
