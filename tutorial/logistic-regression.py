import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 100
vector_set = []

for idx in xrange(num_points):
    x = np.random.normal(0.0, 1)
    y = 1 if x * 0.3 + 0.1 + np.random.normal(0.0, 0.03) > 0 else 0
    vector_set.append([x, y])

x_data = [v[0] for v in vector_set]
y_data = [v[1] for v in vector_set]

# plt.plot(x_data, y_data, 'ro', label='Original data')
# plt.legend()
# plt.show()

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = tf.sigmoid(W * x_data + b)

one = tf.ones(y.get_shape(), dtype=tf.float32)
loss = -tf.reduce_mean(y_data * tf.log(y) + (one-y_data) * tf.log(one-y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print 'params-before-training', sess.run(W), sess.run(b), '\n'

thresholdvec = tf.ones_like(one, dtype=tf.float32) * 0.5
correct_prediction = tf.equal(tf.cast(y_data, tf.int32), tf.cast(tf.greater(y, thresholdvec), tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for step in range(200):
    sess.run(train)

    if step % 10 == 0:
        print 'params', step, sess.run(W)[0], sess.run(b)[0]
        print 'accuracy: ', sess.run(accuracy)
        print 'loss: ', sess.run(loss)
        plt.plot(x_data, y_data, 'ro')
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
        plt.show()

sess.close()