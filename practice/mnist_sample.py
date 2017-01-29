# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

LEARNING_RATE = 0.1     # Gradient Descent algorithm에 적용할 learning rate
TRAINING_EPOCHS = 10    # 총 Training 횟수
BATCH_SIZE = 100        # 전체 Training data set에서 몇개의 data를 가져올지
LOGS_PATH = './.logs'
DATA_PATH = './.data'

# Load input data
mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')    # [total_data_set_size, 28*28 pixels]
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')     # [total_data_set_size, numbers between 0 and 9]

# Set model parameters
with tf.name_scope("weights"):
    W = tf.Variable(tf.zeros([784, 10]))
with tf.name_scope("biases"):
    b = tf.Variable(tf.zeros([10]))

# Construct inference model
with tf.name_scope('softmax'):
    hypothesis = tf.matmul(x, W) + b
    inference = tf.nn.softmax(hypothesis)

# Define cost function : using cross entropy
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(inference), reduction_indices=1))

# Optimizer : Gradient Descent Optimizer
with tf.name_scope('training'):
    learning_rate = tf.constant(LEARNING_RATE)
    train_operation = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=cross_entropy)

# Test model's accuracy
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(inference, 1), tf.argmax(y, 1))    # 예측값 vs 실제값
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))         # bool을 float로 cast한 뒤 평균냄.

# Tensorboard
w_hist = tf.summary.histogram('weights', W)
b_hist = tf.summary.histogram('biases', b)
y_hist = tf.summary.histogram('inference', inference)
cost_summ = tf.summary.scalar('cost', cross_entropy)
accuracy_summ = tf.summary.scalar('accuracy', accuracy_operation)
summary_operation = tf.summary.merge_all()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
writer = tf.summary.FileWriter(LOGS_PATH, tf.get_default_graph())

# Training
for epoch in range(TRAINING_EPOCHS):
    avg_cost = 0.
    batch_count = int(mnist.train.num_examples / BATCH_SIZE)
    for i in range(batch_count):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        _, cost, summary = sess.run(
            [train_operation, cross_entropy, summary_operation],
            feed_dict={x: batch_xs, y: batch_ys}
        )
        avg_cost += cost/batch_count
        writer.add_summary(summary, epoch * batch_count + i)
    print('Epoch: %04d' % (epoch + 1), 'cost={:.9f}'.format(avg_cost))

print('------------------Training Finished------------------')
print('Accuracy:', accuracy_operation.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}, session=sess))
