from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

import iris
import iris_input

from iris_flags import *


def eval_once(saver, summary_writer,
              top_k_op, summary_op,
              weights_vars, dataset,
              images, labels):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[
                -1]
        else:
            print('No checkpoint file found')
            return

        num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        W_fc1_ = np.load(FLAGS.train_dir + '/weights/W_fc1.npy')
        W_fc2_ = np.load(FLAGS.train_dir + '/weights/W_fc2.npy')
        W_fc3_ = np.load(FLAGS.train_dir + '/weights/W_fc3.npy')
        while step < num_iter:
            image_batch, label_batch = dataset.get_batch()
            predictions = sess.run([top_k_op],
                                   feed_dict={weights_vars[0]: W_fc1_,
                                              weights_vars[1]: W_fc2_,
                                              weights_vars[2]: W_fc3_,
                                              images: image_batch,
                                              labels: label_batch})
            true_count += np.sum(predictions)
            step += 1

        # summary_str = sess.run(summary_op, feed_dict={W_fc: W_fc_}) ## NEW ##
        # summary_writer.add_summary(summary_str, step) ## NEW ##

        # Compute precision @ 1.
        precision = true_count / total_sample_count
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    return precision


def evaluate():
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    with tf.Graph().as_default() as g:
        iris_dataset = iris_input.Iris()
        images = tf.placeholder(tf.float32,
                                [FLAGS.batch_size, iris_input.INPUT_SIZE])
        labels = tf.placeholder(tf.int64, [FLAGS.batch_size])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        W_fc1 = tf.placeholder(tf.float32, [iris_input.INPUT_SIZE, 4])
        W_fc2 = tf.placeholder(tf.float32, [4, 4])
        W_fc3 = tf.placeholder(tf.float32, [4, 3])
        weights = [W_fc1, W_fc2, W_fc3]
        logits = iris.inference(images, weights)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        #    variable_averages = tf.train.ExponentialMovingAverage(
        #        mnist.MOVING_AVERAGE_DECAY)
        #    variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            precision = eval_once(saver, summary_writer, top_k_op, summary_op,
                                  weights, iris_dataset, images, labels)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
    return precision


def evaluate_best():
    with tf.Graph().as_default() as g:
        dataset = iris_input.Iris()
        images = tf.placeholder(tf.float32,
                                [FLAGS.batch_size, iris_input.INPUT_SIZE])
        labels = tf.placeholder(tf.int64, [FLAGS.batch_size])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        W_fc1 = tf.placeholder(tf.float32, [iris_input.INPUT_SIZE, 4])
        W_fc2 = tf.placeholder(tf.float32, [4, 4])
        W_fc3 = tf.placeholder(tf.float32, [4, 3])
        weights = [W_fc1, W_fc2, W_fc3]
        logits = iris.inference(images, weights)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt_path = FLAGS.train_dir + '/best_weights/model.ckpt-79'
            saver.restore(sess, ckpt_path)

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            W_fc1_ = np.load(FLAGS.train_dir + '/best_weights/W_fc1.npy')
            W_fc2_ = np.load(FLAGS.train_dir + '/best_weights/W_fc2.npy')
            W_fc3_ = np.load(FLAGS.train_dir + '/best_weights/W_fc3.npy')
            while step < num_iter:
                image_batch, label_batch = dataset.get_batch()
                predictions = sess.run([top_k_op],
                                       feed_dict={W_fc1: W_fc1_,
                                                  W_fc2: W_fc2_,
                                                  W_fc3: W_fc3_,
                                                  images: image_batch,
                                                  labels: label_batch})
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count

        print(precision)


def main(argv=None):  # pylint: disable=unused-argument
    # mnist.maybe_download_and_extract()
    # if tf.gfile.Exists(FLAGS.eval_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    # tf.gfile.MakeDirs(FLAGS.eval_dir)
    # evaluate()
    evaluate_best()


if __name__ == '__main__':
    tf.app.run()
