import tensorflow as tf

input1 = tf.constant([[1, 2]])
input2 = tf.constant([[2],[3]])

product1 = tf.matmul(input1, input2)
product2 = tf.matmul(input2, input1)
product3 = tf.mul(input1, input1)

with tf.Session() as sess:
    print sess.run(product1)
    print sess.run(product2)
    print sess.run(product3)