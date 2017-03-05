import tensorflow as tf

hello = tf.constant('hello world')

sess = tf.Session()

result = sess.run(hello)

print(result)