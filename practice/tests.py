from django.test import TestCase

from practice.services import MNIST


class MNISTTestCase(TestCase):

    def testWholeOperation(self):
        mnist = MNIST()

        mnist.load_training_data()
        assert mnist.training_data is not None

        mnist.set_algorithm()
        assert mnist.hypothesis is not None
        assert mnist.inference is not None
        assert mnist.cost is not None

        mnist.set_training(0.1, 10)
        assert mnist.learning_rate is not None
        assert mnist.optimizer is not None
        assert mnist.training_epochs is not None
        assert mnist.batch_size is not None
        assert mnist.train_operation is not None
        assert mnist.accuracy_operation is not None
        assert mnist.summary_operation is not None

        mnist.run()

        mnist.load_testing_data()
        assert mnist.test_data is not None

        mnist.test()
