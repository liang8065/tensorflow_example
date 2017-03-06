import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
flags.DEFINE_integer('max_steps', 2001, 'Number of steps to run trainer')
flags.DEFINE_integer('hidden1_unit', 128, 'Number of units in hidden layer 1')
flags.DEFINE_integer('hidden2_unit', 32, 'Number of units in hidden layer 2')
flags.DEFINE_integer('batch_size', 100, 'Batch size.'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'train_dir/', 'Directory to put the training data')
flags.DEFINE_boolean('fake_data', False, 'If true, use fake data '
                     'for unit testing')
import mnist
import time
from tensorflow.examples.tutorials.mnist import input_data

def placeholder_inputs(batch_size):
    image_placeholder = tf.placeholder(tf.float32, shape=[batch_size, mnist.IMAGE_PIXELS])
    label_placeholder = tf.placeholder(tf.int32, shape=batch_size)

    return image_placeholder, label_placeholder


def fill_placeholder(data_set, images_pl, labels_pl):
    image_feed, label_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {images_pl: image_feed, labels_pl: label_feed}

    return feed_dict


def do_eval(sess,
            eval_correct,
            image_placeholder,
            label_placeholder,
            data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size

    for step in range(steps_per_epoch):
        feed_dict = fill_placeholder(data_set, image_placeholder, label_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)

    precision = 1.0 * true_count / num_examples
    print('Number of examples: %d Num correct: %d Precision: @ 1 : %.4f' % (num_examples, true_count, precision))


def run_training():
    data_sets = input_data.read_data_sets('MNIST_data', FLAGS.fake_data)

    with tf.Graph().as_default():
        image_placeholder, label_placeholder = placeholder_inputs(FLAGS.batch_size)

        logits = mnist.inference(image_placeholder, FLAGS.hidden1_unit, FLAGS.hidden2_unit)

        loss = mnist.loss(logits, label_placeholder)

        train_op = mnist.training(loss, FLAGS.learning_rate)

        eval_correct = mnist.evaluation(logits, label_placeholder)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in range(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = fill_placeholder(data_sets.train, image_placeholder, label_placeholder)

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 100 == 0 :
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

                summary_str = sess.run(summary_op, feed_dict = feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if step % 1000 == 0:
                saver.save(sess, FLAGS.train_dir, global_step=step)

                print('Training Data Eval: ')
                do_eval(sess, eval_correct, image_placeholder, label_placeholder, data_sets.train)

                print('Validation Data Eval: ')
                do_eval(sess, eval_correct, image_placeholder, label_placeholder, data_sets.validation)

                print('Test Data Eval:')
                do_eval(sess, eval_correct, image_placeholder, label_placeholder, data_sets.test)


def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()

