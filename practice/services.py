from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

from EasyTensor.redis_utils import RedisManager


class BasePractice(object):
    # Path는 각 practice 별로 정의한다.
    LOGS_PATH = None
    DATA_PATH = None

    def load_training_data(self, *params):
        raise NotImplementedError()

    @staticmethod
    def get_algorithm_settings():
        raise NotImplementedError()

    def set_algorithm(self, *params):
        raise NotImplementedError()

    @staticmethod
    def get_training_settings():
        raise NotImplementedError()

    def set_training(self, *params):
        raise NotImplementedError()

    def run(self, *params):
        raise NotImplementedError()

    def load_testing_data(self, *params):
        raise NotImplementedError()

    def test(self, *params):
        raise NotImplementedError()

    @staticmethod
    def tensorboard():
        raise NotImplementedError()

class MNIST(BasePractice):

    LOGS_PATH = 'practice/.logs'
    DATA_PATH = 'practice/.data'

    # Data
    training_data = None
    test_data = None
    X = None
    Y = None

    # Algorithm
    hypothesis = None
    inference = None
    cost = None

    # Training
    learning_rate = None
    optimizer = None
    training_epochs = None
    batch_size = None

    # Operation
    train_operation = None
    accuracy_operation = None
    summary_operation = None

    def load_training_data(self, *params):
        dataset = input_data.read_data_sets(self.DATA_PATH, one_hot=True)
        self.training_data = dataset.train
        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, [None, 784], name='x-input')  # [total_data_set_size, 28*28 pixels]
            self.Y = tf.placeholder(tf.float32, [None, 10], name='y-input')  # [total_data_set_size, numbers between 0 and 9]

    @staticmethod
    def get_algorithm_settings():
        return {
            'Num of layers': [
                1, 2, 3
            ],
            'Activation Function': [
                'Sigmoid', 'ReLU'
            ],
            'Optimizer': [
                'GradientDescentOptimizer', 'AdamOptimizer'
            ],
            'Weight Initialization': [
                'No', 'Yes'
            ],
            'Dropout': [
                'No', 'Yes'
            ]
        }

    def set_algorithm(self, *params):
        with tf.name_scope("weights"):
            W = tf.Variable(tf.zeros([784, 10]))
        with tf.name_scope("biases"):
            b = tf.Variable(tf.zeros([10]))
        # Construct inference model
        with tf.name_scope('softmax'):
            self.hypothesis = tf.matmul(self.X, W) + b
            self.inference = tf.nn.softmax(self.hypothesis)
        # Define cost function : using cross entropy
        with tf.name_scope('cross_entropy'):
            self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(tf.clip_by_value(self.inference, 1e-10, 1.0)), reduction_indices=1))
        # Tensorboard
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("inference", self.inference)
        tf.summary.scalar("cost", self.cost)

    @staticmethod
    def get_training_settings():
        return {
            'Learning Rate': 0.1,
            'Optimization Epoch': 10,
        }

    def set_training(self, *params):
        def select_optimizer(optimizer, rate):
            if optimizer == 'GradientDescentOptimizer':
                return tf.train.GradientDescentOptimizer(learning_rate=rate)
            elif optimizer == 'AdamOptimizer':
                return tf.train.AdamOptimizer(learning_rate=rate)
        self.learning_rate = tf.constant(params[1])
        with tf.name_scope('training'):
            self.optimizer = select_optimizer(params[0], self.learning_rate)
            self.train_operation = self.optimizer.minimize(loss=self.cost)
        self.training_epochs = params[2]
        self.batch_size = 100
        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.inference, 1), tf.argmax(self.Y, 1))
            self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy_operation)
        self.summary_operation = tf.summary.merge_all()

    def run(self, *params):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(self.LOGS_PATH, tf.get_default_graph())
        for epoch in range(self.training_epochs):
            avg_cost = 0.
            batch_count = int(self.training_data.num_examples / self.batch_size)
            for i in range(batch_count):
                batch_xs, batch_ys = self.training_data.next_batch(self.batch_size)
                _, cost, summary = self.sess.run(
                    [self.train_operation, self.cost, self.summary_operation],
                    feed_dict={self.X: batch_xs, self.Y: batch_ys}
                )
                avg_cost += cost / batch_count
                writer.add_summary(summary, epoch * batch_count + i)
            message = 'Epoch %03d : cost=%.9f' % (epoch + 1, avg_cost)
            RedisManager.set_message('mnist', message)
            print(message)

    def load_testing_data(self, *params):
        dataset = input_data.read_data_sets(self.DATA_PATH, one_hot=True)
        self.test_data = dataset.test

    def test(self, *params):
        print('Accuracy:', self.accuracy_operation.eval(
            feed_dict={self.X: self.test_data.images, self.Y: self.test_data.labels},
            session=self.sess
        ))

    @staticmethod
    def tensorboard():
        path = "tensorboard --logdir=" + os.path.abspath('./' + MNIST.LOGS_PATH) + " &"
        os.system(path)
