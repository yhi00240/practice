from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

class BasePractice(object):
    LOGS_PATH = './.logs'
    DATA_PATH = './.data'

    def loadTrainingData(self, *params):
        raise NotImplementedError()

    def setAlgorithm(self, *params):
        raise NotImplementedError()

    def setTraining(self, *params):
        raise NotImplementedError()

    def run(self, *params):
        raise NotImplementedError()

    def getStatus(self, *params):
        raise NotImplementedError()

class MNIST(BasePractice):
    # Data
    training_data = None
    test_data = None
    X = None
    Y = None

    # Algorithm
    hypothesis = None
    inference = None
    cost = None
    optimizer = None

    # Training
    learning_rate = None
    training_epochs = None
    batch_size = None

    # Operation
    train_operation = None
    summary_operation = None
    accuracy_operation = None

    def loadTrainingData(self, *params):
        mnist = input_data.read_data_sets(self.DATA_PATH, one_hot=True)
        self.training_data = mnist.train
        self.test_data = mnist.test
        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, [None, 784], name='x-input')  # [total_data_set_size, 28*28 pixels]
            self.Y = tf.placeholder(tf.float32, [None, 10], name='y-input')  # [total_data_set_size, numbers between 0 and 9]

    def setAlgorithm(self, *params):
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
            self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.inference), reduction_indices=1))
        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.inference, 1), tf.argmax(self.Y, 1))  # 예측값 vs 실제값
            self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # bool을 float로 cast한 뒤 평균냄.

        # Tensorboard
        w_hist = tf.summary.histogram('weights', W)
        b_hist = tf.summary.histogram('biases', b)
        y_hist = tf.summary.histogram('inference', self.inference)
        cost_summ = tf.summary.scalar('cost', self.cost)
        accuracy_summ = tf.summary.scalar('accuracy', self.accuracy_operation)
        self.summary_operation = tf.summary.merge_all()

    def setTraining(self, *params):
        self.learning_rate = tf.constant(0.1)
        with tf.name_scope('training'):
            self.train_operation = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss=self.cost)
        self.training_epochs = 10
        self.batch_size = 100

    def run(self):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter(self.LOGS_PATH, tf.get_default_graph())
        for epoch in range(self.training_epochs):
            avg_cost = 0.
            batch_count = int(self.training_data.num_examples / self.batch_size)
            for i in range(batch_count):
                batch_xs, batch_ys = self.training_data.next_batch(self.batch_size)
                _, cost, summary = sess.run(
                    [self.train_operation, self.cost, self.summary_operation],
                    feed_dict={self.X: batch_xs, self.Y: batch_ys}
                )
                avg_cost += cost / batch_count
                writer.add_summary(summary, epoch * batch_count + i)
            print('Epoch: %04d' % (epoch + 1), 'cost={:.9f}'.format(avg_cost))

    def getStatus(self):
        # TODO : training status 어떻게 리턴시킬지
        pass