import tensorflow as tf
import math

NUM_CLASSES = 10

IMAGE_SIZE = 28

IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(image, hidden1_unit, hidden2_unit):

    with tf.name_scope("hidden1"):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_unit], stddev=1.0 / math.sqrt(IMAGE_PIXELS)), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_unit]))
        hidden1 = tf.nn.relu(tf.matmul(image, weights) + biases)

    with tf.name_scope("hidden2"):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_unit, hidden2_unit], stddev=1.0 / math.sqrt(hidden1_unit)), name='weights')
        biases = tf.Variable(tf.zeros([hidden2_unit]))
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope("softmax_linear"):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_unit, NUM_CLASSES], stddev=1.0 / math.sqrt(hidden2_unit)), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]))
        logits = tf.matmul(hidden2, weights) + biases

    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    cur_loss = tf.reduce_mean(cross_entropy, name='xentropy-mean')

    return cur_loss


def training(cur_loss, learning_rate):
    tf.summary.scalar(cur_loss.op.name, cur_loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(cur_loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
