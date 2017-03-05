import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)

add = tf.add(input1, input2)

sess = tf.Session()

result = sess.run(add)
print result

sess.close()

#with tf.Session() as sess:
#    result = sess.run(add)
#    print result