import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)
update = tf.assign(state, tf.add(state, one))

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(state)

    for _ in xrange(3):
        sess.run(update)
        print sess.run(state)
        print sess.run([update, update, state])