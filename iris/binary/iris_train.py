from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import iris
import iris_input
import iris_eval
import os

from iris_flags import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Learning rate decay factor.
LEARNING_RATE_DECAY_FACTOR = 0.316227766  # This is sqrt(1/10)


def save_training_params():
    f = open(FLAGS.train_dir + '/summary.txt', 'w')
    f.write('Epochs = ' + str(FLAGS.epochs) + '\n')
    f.write('Batch size = ' + str(FLAGS.batch_size * FLAGS.num_gpus) + '\n')
    f.write('Learning rate = ' + str(FLAGS.learning_rate) + '\n')
    f.write('Probabilities weight decay = ' + str(FLAGS.wd) + '\n')
    f.write('Weight decay = ' + str(FLAGS.wd_weights) + '\n')
    f.write('Dropout rate = ' + str(FLAGS.dropout) + '\n')
    f.write('Epochs per LR decay = ' + str(FLAGS.lr_decay_epochs) + '\n')
    f.write('First layer ternary = ' + str(FLAGS.first_layer_ternary) + '\n\n')
    f.close()


def tower_loss(scope, images, labels):
    # Build inference Graph.
    logits = iris.inference(images, [None, None, None], train=True)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = iris.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % iris.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (iris.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 (FLAGS.batch_size * FLAGS.num_gpus))
        decay_steps = int(num_batches_per_epoch * FLAGS.lr_decay_epochs)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(lr)

        # Get images and labels for iris.
        iris_dataset = iris_input.Iris()
        images = tf.placeholder(tf.float32, [FLAGS.batch_size,
                                             iris_input.INPUT_SIZE])
        labels = tf.placeholder(tf.int16, [FLAGS.batch_size])
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope(
                        '%s_%d' % (iris.TOWER_NAME, i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        loss = tower_loss(scope, images, labels)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                      scope)

                        # Added for BN - 25.7.17 Oran
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            iris.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        #    train_op = tf.group(variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        step_reached = -1
        max_steps = int(
            FLAGS.epochs * iris.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / (
                FLAGS.batch_size * FLAGS.num_gpus))
        best_precision = 0
        f = open(FLAGS.train_dir + '/summary.txt', 'a')
        # Load model if not a new run
        if FLAGS.new_run == False:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                step_reached = \
                ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        for step in xrange(int(step_reached) + 1, max_steps):
            start_time = time.time()
            image_batch, label_batch = iris_dataset.get_batch()
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict={images: image_batch,
                                                labels: label_batch})
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = (
                '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op,
                                       feed_dict={images: image_batch,
                                                  labels: label_batch})
                summary_writer.add_summary(summary_str, step)

            if (step % 400 == 0) or (step + 1) == max_steps:
                # Save the model checkpoint periodically.
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                precision = []
                for i in xrange(7):
                    iris.draw_weights(sess)
                    precision += [iris_eval.evaluate()]
                    if precision[i] > best_precision:
                        best_precision = precision[i]
                        os.system(
                            'cp ' + FLAGS.train_dir + '/weights/* ' + FLAGS.train_dir + '/best_weights/')
                        os.system(
                            'rm ' + FLAGS.train_dir + '/best_weights/model.ckpt*'
                        )
                        os.system(
                            'cp ' + os.path.join(FLAGS.train_dir, 'model.ckpt-{}*'.format(step))
                            + ' ' + FLAGS.train_dir + '/best_weights/'
                        )
                print('Average precision: ' + str(round(np.mean(precision), 3)))
                f.write('step: ' + str(step) + ', average precision = ' + str(
                    round(np.mean(precision), 3)) + '\n')

        f.write('best precision = ' + str(round(best_precision, 3)) + '\n')
        print('Best precision: ' + str(round(best_precision, 3)))
        f.close()


def main(argv=None):  # pylint: disable=unused-argument
    if FLAGS.new_run:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir + '/weights')
        tf.gfile.MakeDirs(FLAGS.train_dir + '/best_weights')
        save_training_params()
    train()


if __name__ == '__main__':
    tf.app.run()