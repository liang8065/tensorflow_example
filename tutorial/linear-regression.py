import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 1000
vector_sets = []

for idx in range(num_points):
    x = np.random.normal(0.0, 0.55)
    y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vector_sets.append([x, y])

x_data = [v[0] for v in vector_sets]
y_data = [v[1] for v in vector_sets]

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.show()

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print('params-before-training', sess.run(W), sess.run(b), '\n')

for step in range(20):
    sess.run(train)
    cur_w = sess.run(W)
    cur_b = sess.run(b)

    print('loss', (step, sess.run(loss)))
    print('params', step, cur_w, cur_b, '\n')
    labelstr = 'training step = ', step
    plt.plot(x_data, y_data, 'ro', label=labelstr)
    plt.plot(x_data, cur_w * x_data + cur_b)
    plt.legend()
    plt.show()

sess.close()