import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def weights_variable(shape):
    initial_val = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_val)


def bias_variable(shape):
    initial_val = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_val)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def maxpooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])

w_conv1 = weights_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)    # 28 * 28 * 32
h_pool1 = maxpooling_2x2(h_conv1)   # 14 * 14 * 32

w_conv2 = weights_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)    # 14 * 14 * 64
h_pool2 = maxpooling_2x2(h_conv2)   # 7 * 7 * 64

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
w_fc1 = weights_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weights_variable([1024, 10])
b_fc2 = bias_variable([10])
h_fc2 = tf.nn.softmax(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(h_fc2), reduction_indices=1))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cross_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(h_fc2, 1))
accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

for i in xrange(2001):
    images, labels = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: images, ys: labels, keep_prob: 0.5})

    if i % 50 == 0:
        print "train accuracy: ", sess.run(accuracy, feed_dict={xs: images, ys: labels, keep_prob: 1.0})
        # print "test accuracy: ", sess.run(accuracy, feed_dict={X: data_sets.test.images, y: data_sets.test.labels})
        # print "loss: ", sess.run(cross_entropy, feed_dict={xs: images, ys:labels, keep_prob:0.5})

print "\nfinal test accuracy: ", sess.run(accuracy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0})
