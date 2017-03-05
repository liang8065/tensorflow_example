import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_sets = input_data.read_data_sets('MNIST_data', one_hot=True)

print data_sets.train.images.shape
print data_sets.train.labels.shape

X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_ = tf.nn.softmax(tf.matmul(X, W) + b)
cross_entropy = - tf.reduce_sum(y * tf.log(y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

cross_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)

    for i in xrange(1000):
        images, labels = data_sets.train.next_batch(100)
        sess.run([train_step, cross_entropy], feed_dict={X:images, y:labels})
        if i % 20 == 0:
            #print "train accuracy: ", sess.run(accuracy, feed_dict={X: images, y:labels})
            #print "test accuracy: ", sess.run(accuracy, feed_dict={X: data_sets.test.images, y: data_sets.test.labels})
            print "loss: ", sess.run(cross_entropy, feed_dict={X: images, y:labels})

    print "\nfinal test accuracy: ", sess.run(accuracy, feed_dict={X: data_sets.test.images, y: data_sets.test.labels})