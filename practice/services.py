# -*- coding:utf-8 -*-
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

from EasyTensor.redis_utils import RedisManager
from practice import high_accuracy
from practice.utils import image_to_mnist

class BasePractice(object):
    # Path는 각 practice 별로 정의한다.
    LOGS_PATH = None
    DATA_PATH = None

    def load_data(self, *params):
        raise NotImplementedError()

    def set_algorithm(self, *params):
        raise NotImplementedError()

    def set_training(self, *params):
        raise NotImplementedError()

    def run(self, *params):
        raise NotImplementedError()

    def test_all(self, *params):
        raise NotImplementedError()

    def test_single(self, *params):
        raise NotImplementedError()

    @staticmethod
    def tensorboard():
        raise NotImplementedError()

class MNIST(BasePractice):

    # Static Variables
    LOGS_PATH = 'practice/.logs'
    DATA_PATH = 'practice/.data'
    MODEL_PATH = 'practice/.models/mnist'
    BATCH_SIZE = 100
    IMAGE_SIZE = 28
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
    NUM_CLASSES = 10
    ONE_BYTE = 255
    X = None
    Y = None
    DROPOUT_RATE = None

    def __init__(self):
        self.sess = tf.Session()
        self.data = None
        self.hypothesis = None
        self.cost = None
        self.optimizer = None
        self.training_epochs = None
        self.train_operation = None
        self.accuracy_operation = None
        self.summary_operation = None
        self.save_path = None

        with tf.name_scope('Input'):
            MNIST.X = tf.placeholder(tf.float32, [None, MNIST.IMAGE_PIXELS], name='images')  # [total_data_set_size, 28*28 pixels]
            MNIST.Y = tf.placeholder(tf.float32, [None, MNIST.NUM_CLASSES], name='labels')  # [total_data_set_size, numbers between 0 and 9]
        MNIST.DROPOUT_RATE = tf.placeholder(tf.float32)

    def load_data(self, test=False, *params):
        data_set = input_data.read_data_sets(MNIST.DATA_PATH, one_hot=True)
        if test:
            self.data = data_set.test
        else:
            self.data = data_set.train

    @staticmethod
    def single_layer(weight_initialize):
        if weight_initialize:
            W = tf.get_variable('weights', shape=[MNIST.IMAGE_PIXELS, MNIST.NUM_CLASSES], initializer=xavier_initializer())
            b = tf.Variable(tf.zeros([MNIST.NUM_CLASSES]), name='biases')
        else:
            W = tf.Variable(tf.random_normal([MNIST.IMAGE_PIXELS, MNIST.NUM_CLASSES]), name='weights')
            b = tf.Variable(tf.random_normal([MNIST.NUM_CLASSES]), name='biases')
        hypothesis = tf.add(tf.matmul(MNIST.X, W), b)
        return hypothesis, [W, b]

    @staticmethod
    def multi_layer(weight_initialize, sigmoid, dropout):
        if weight_initialize:
            W1 = tf.get_variable('W1', shape=[MNIST.IMAGE_PIXELS, 256], initializer=xavier_initializer())
            W2 = tf.get_variable('W2', shape=[256, 256], initializer=xavier_initializer())
            W3 = tf.get_variable('W3', shape=[256, MNIST.NUM_CLASSES], initializer=xavier_initializer())
            B1 = tf.Variable(tf.random_normal([256]), name='B1')
            B2 = tf.Variable(tf.random_normal([256]), name='B2')
            B3 = tf.Variable(tf.random_normal([MNIST.NUM_CLASSES]), name='B3')
        else:
            W1 = tf.Variable(tf.random_normal([MNIST.IMAGE_PIXELS, 256]), name='W1')
            W2 = tf.Variable(tf.random_normal([256, 256]), name='W2')
            W3 = tf.Variable(tf.random_normal([256, MNIST.NUM_CLASSES]), name='W3')
            B1 = tf.Variable(tf.random_normal([256]), name='B1')
            B2 = tf.Variable(tf.random_normal([256]), name='B2')
            B3 = tf.Variable(tf.random_normal([MNIST.NUM_CLASSES]), name='B3')
        activation_function = tf.nn.sigmoid if sigmoid else tf.nn.relu
        if dropout:
            _L1 = activation_function(tf.add(tf.matmul(MNIST.X, W1), B1), name='Hidden_layer1')
            L1 = tf.nn.dropout(_L1, MNIST.DROPOUT_RATE, name='Hidden_dropout_layer1')
            _L2 = activation_function(tf.add(tf.matmul(L1, W2), B2), name='Hidden_layer2')
            L2 = tf.nn.dropout(_L2, MNIST.DROPOUT_RATE, name='Hidden_dropout_layer2')
            hypothesis = tf.add(tf.matmul(L2, W3), B3)

        else:
            L1 = activation_function(tf.add(tf.matmul(MNIST.X, W1), B1), name='Hidden_layer1')
            L2 = activation_function(tf.add(tf.matmul(L1, W2), B2), name='Hidden_layer2')
            hypothesis = tf.add(tf.matmul(L2, W3), B3)

        return hypothesis, [W1, W2, W3, B1, B2, B3]

    def get_model(self, model_type, weight_initialize, activation_function, dropout):
        single_layer = (model_type == 'Single layer')
        initialize = (weight_initialize == 'Yes')
        sigmoid = (activation_function == 'Sigmoid')
        use_dropout = (dropout == 'Yes')
        print('[model type] single_layer:{0}, initialize:{1}, function:{2}, dropout:{3}'.format(single_layer, initialize, sigmoid, use_dropout))
        # Define model saving path
        save_path = MNIST.MODEL_PATH
        save_path += '_single' if single_layer else '_multi'
        save_path += '_w-init' if initialize else '_w-random'
        save_path += '_sigmoid' if sigmoid else '_relu'
        save_path += '_dropout' if use_dropout else '_connect_all'
        if single_layer:
            hypothesis, variable = MNIST.single_layer(initialize)
        else:
            hypothesis, variable = MNIST.multi_layer(initialize, sigmoid, use_dropout)
        return save_path, hypothesis, variable

    def set_algorithm(self, model_type, weight_initialize, activation_function, dropout):
        self.save_path, self.hypothesis, variables = self.get_model(model_type, weight_initialize, activation_function, dropout)
        with tf.name_scope('Cost'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.hypothesis, MNIST.Y))
        tf.summary.scalar('cost', self.cost)

    def set_training(self, optimizer, learning_rate, epochs):
        def select_optimizer(optimizer, learning_rate):
            if optimizer == 'GradientDescentOptimizer':
                return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            elif optimizer == 'AdamOptimizer':
                return tf.train.AdamOptimizer(learning_rate=learning_rate)
            # else
            # TODO else일경우 오류처리
        with tf.name_scope('Training'):
            self.optimizer = select_optimizer(optimizer, learning_rate)
            self.train_operation = self.optimizer.minimize(loss=self.cost, name='optimizer')
        self.training_epochs = epochs
        with tf.name_scope('Test'):
            correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(MNIST.Y, 1))
            self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy_operation)
        self.summary_operation = tf.summary.merge_all()

    def run(self, *params):
        self.sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(self.LOGS_PATH, tf.get_default_graph())
        for epoch in range(self.training_epochs):
            avg_cost = 0.
            batch_count = int(self.data.num_examples / MNIST.BATCH_SIZE)
            for i in range(batch_count):
                batch_xs, batch_ys = self.data.next_batch(MNIST.BATCH_SIZE)
                _, cost, summary = self.sess.run(
                    [self.train_operation, self.cost, self.summary_operation],
                    feed_dict={MNIST.X: batch_xs, MNIST.Y: batch_ys, MNIST.DROPOUT_RATE: 0.7}
                )
                avg_cost += cost / batch_count
                writer.add_summary(summary, epoch * batch_count + i)
            message = 'Epoch %03d : cost=%.9f' % (epoch + 1, avg_cost)
            RedisManager.set_message('mnist', message)
            print(message)
        saver = tf.train.Saver()
        saver.save(self.sess, self.save_path)
        print('Model saved in file: {0}'.format(self.save_path))

    def test_all(self, model_type, weight_initialize, activation_function, dropout):
        save_path, hypothesis, variables = self.get_model(model_type, weight_initialize, activation_function, dropout)
        saver = tf.train.Saver(variables)
        saver.restore(self.sess, save_path)
        print('Model restored from file: {0}'.format(save_path))
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(MNIST.Y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy = accuracy_operation.eval(
            feed_dict={MNIST.X: self.data.images, MNIST.Y: self.data.labels, MNIST.DROPOUT_RATE: 1},
            session=self.sess
        )
        print('Accuracy', accuracy)
        return accuracy

    def test_single(self, image_data, model_type, weight_initialize, activation_function, dropout):
        save_path, hypothesis, variables = self.get_model(model_type, weight_initialize, activation_function, dropout)
        y = tf.nn.softmax(hypothesis)
        saver = tf.train.Saver(variables)
        saver.restore(self.sess, save_path)
        print('Model restored from file: {0}'.format(save_path))
        y_ref, variables = high_accuracy.convolutional(MNIST.X, 1.0)
        saver = tf.train.Saver(variables)
        saver.restore(self.sess, "practice/trained_model/convolutional.ckpt")
        mnist_data = image_to_mnist(image_data)
        return self.sess.run(y, feed_dict={MNIST.X: mnist_data, MNIST.DROPOUT_RATE: 1}).flatten().tolist(), self.sess.run(y_ref, feed_dict={MNIST.X: mnist_data, MNIST.DROPOUT_RATE: 1}).flatten().tolist()

    @staticmethod
    def tensorboard():
        path = "tensorboard --logdir=" + os.path.abspath('./' + MNIST.LOGS_PATH) + " &"
        os.system(path)
