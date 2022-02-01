from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf
import numpy as np

import iris_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 150,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('hot_start_dir',
                           os.path.expanduser('~/Downloads/LRnets/iris_full_precision'),
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_float('wd', 1e-6,
                          """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('wd_weights', 0.0001,
                          """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('dropout', 0.5, """Dropout rate.""")
tf.app.flags.DEFINE_boolean('hot_start', False,
                            """Whether this is a new run or not.""")
tf.app.flags.DEFINE_boolean('first_layer_ternary', True,
                            """Whether the first layer  or not.""")

INPUT_SIZE = iris_input.INPUT_SIZE
NUM_CLASSES = iris_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = iris_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = iris_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Either BINARY or TERNARY
WEIGHT_TYPE = 'TERNARY'


def draw_ternary_weight(a, b=None):
    if WEIGHT_TYPE == 'BINARY':
        var = 2 * (np.random.binomial(1, p=a) - 0.5)
    elif WEIGHT_TYPE == 'TERNARY':
        var = ((2 * (np.random.binomial(1, p=a / (a + b)) - 0.5))
               * np.random.binomial(1, p=a + b))
    return var


def init_values_hs_ver3(W):
    if WEIGHT_TYPE == 'BINARY':
        W = np.array(W)
        W[W > 1] = 1
        W[W < -1] = -1
        a_init = 0.5 * (1 + W)
        a_init[a_init > 0.95] = 0.95
        a_init[a_init < 0.05] = 0.05
        a_init = init_probs(a_init)
        return a_init
    elif WEIGHT_TYPE == 'TERNARY':
        W = np.array(W)
        W[W > 1] = 1
        W[W < -1] = -1
        c_init = 0.95 - 0.9 * np.abs(W)
        a_nz_init = 0.5 * (1 + W / (1 - c_init))
        a_nz_init[a_nz_init > 0.95] = 0.95
        a_nz_init[a_nz_init < 0.05] = 0.05
        #    a_nz_init = np.float32(W>=0.33) + 0.5*((W<0.33) & (W>-0.33))
        c_init = init_probs(c_init)
        a_nz_init = init_probs(a_nz_init)
        return c_init, a_nz_init


def init_probs(prob):
    init_val = np.array(prob, dtype=np.float32)
    init_val[init_val < 1e-4] = 1e-4
    init_val[init_val > 0.9999] = 0.9999
    init_val = np.log(init_val / (1 - init_val))
    return init_val


def initializer(scope, shape, prob):
    if WEIGHT_TYPE == 'BINARY':
        if FLAGS.hot_start:
            a_init = init_values_hs_ver3(
                np.load(FLAGS.hot_start_dir + '/W_' + scope.name + '.npy'))
        else:
            a_init = init_probs(np.random.uniform(0.5, 0.5, shape))
        return a_init
    elif WEIGHT_TYPE == 'TERNARY':
        if FLAGS.hot_start:
            c_init, a_nz_init = init_values_hs_ver3(
                np.load(FLAGS.hot_start_dir + '/W_' + scope.name + '.npy'))
        else:
            c_init = init_probs(np.random.uniform(0.45, 0.55, shape))
            a_nz_init = init_probs(
                np.random.binomial(1, p=0.5 * np.ones(shape)))
        if prob == 'c':
            return c_init
        else:
            return a_nz_init


def reparametrization(prev_layer, shape, scope, kernel, conv=True, train=True,
      activation=True):
    if WEIGHT_TYPE == 'BINARY':
        a_ = tf.get_variable('a', initializer=initializer(scope, shape, 'a'),
                             dtype=tf.float32)
        #  a_ = tf.get_variable('a_nz', shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False), dtype=tf.float32)

        a = tf.nn.sigmoid(a_)
        b = 1 - a
        beta_loss = tf.multiply(tf.reduce_sum(tf.multiply(a, 1 - a)), FLAGS.wd,
                                name='beta_loss_a')
        tf.add_to_collection('losses', beta_loss)
        if train:
            mu = a - b
            var = a + b - tf.square(mu)
            normal_dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
            if conv:
                mu_bar = tf.nn.conv2d(prev_layer, mu, [1, 1, 1, 1], padding='SAME')
                sigma_bar = tf.sqrt(
                    tf.nn.conv2d(tf.square(prev_layer), var, [1, 1, 1, 1],
                                 padding='SAME') + 0.001)
            else:
                mu_bar = tf.matmul(prev_layer, mu)
                sigma_bar = tf.sqrt(tf.matmul(tf.square(prev_layer), var) + 0.001)
            res = normal_dist.sample(tf.shape(mu_bar)) * sigma_bar + mu_bar
            tf.summary.histogram('a', a)
            tf.summary.histogram('b', b)
        else:
            if conv:
                res = tf.nn.conv2d(prev_layer, kernel, [1, 1, 1, 1], padding='SAME')
            else:
                res = tf.matmul(prev_layer, kernel)
        res = tf.contrib.layers.batch_norm(
            res, center=True, scale=True, is_training=train)
        if activation:
            res = tf.nn.relu(res)
        return res
    elif WEIGHT_TYPE == 'TERNARY':
        c_ = tf.get_variable('c', initializer=initializer(scope, shape, 'c'), dtype=tf.float32)
        a_nz_ = tf.get_variable('a_nz', initializer=initializer(scope, shape, 'a_nz'), dtype=tf.float32)
        #  c_ = tf.get_variable('c', shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False), dtype=tf.float32)
        #  a_nz_ = tf.get_variable('a_nz', shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False), dtype=tf.float32)
        wd_c_ = tf.multiply(tf.nn.l2_loss(c_), FLAGS.wd, name='weight_loss_c')
        tf.add_to_collection('losses', wd_c_)
        wd_a_nz_ = tf.multiply(tf.nn.l2_loss(a_nz_), FLAGS.wd, name='weight_loss_a_nz')
        tf.add_to_collection('losses', wd_a_nz_)

        c = tf.nn.sigmoid(c_)
        a_nz = tf.nn.sigmoid(a_nz_)
        a = a_nz*(1-c)
        b = (1-a_nz)*(1-c)
        if train:
          mu = a - b
          var = a + b - tf.square(mu)
          normal_dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
          if conv:
            mu_bar = tf.nn.conv2d(prev_layer, mu,  [1, 1, 1, 1], padding='SAME')
            sigma_bar = tf.sqrt(tf.nn.conv2d(tf.square(prev_layer), var,  [1, 1, 1, 1], padding='SAME')+0.001)
          else:
            mu_bar = tf.matmul(prev_layer, mu)
            sigma_bar = tf.sqrt(tf.matmul(tf.square(prev_layer), var)+0.001)
          res = normal_dist.sample(tf.shape(mu_bar))*sigma_bar + mu_bar
          tf.summary.histogram('a',a)
          tf.summary.histogram('b',b)
          tf.summary.histogram('c',c)
        else:
          if conv:
              res = tf.nn.conv2d(prev_layer, kernel, [1, 1, 1, 1], padding='SAME')
          else:
              res = tf.matmul(prev_layer, kernel)
        res = tf.contrib.layers.batch_norm(
            res, center=True, scale=True, is_training=train)
        if activation:
            res = tf.nn.relu(res)
        return res


def get_probs():
    if WEIGHT_TYPE == 'BINARY':
        a_ = tf.get_variable('a')
        a = tf.nn.sigmoid(a_)
        return a
    elif WEIGHT_TYPE == 'TERNARY':
        c_ = tf.get_variable('c')
        a_nz_ = tf.get_variable('a_nz')
        c = tf.nn.sigmoid(c_)
        a_nz = tf.nn.sigmoid(a_nz_)
        a = a_nz * (1 - c)
        b = (1 - a_nz) * (1 - c)
        return a, b


def conv_relu(scope, prev_layer, conv_shape, train):
    kernel = _variable_with_weight_decay(scope.name, shape=conv_shape,
                                         wd=FLAGS.wd_weights)
    conv = tf.nn.conv2d(prev_layer, kernel, [1, 1, 1, 1], padding='SAME')
    conv_normed = tf.contrib.layers.batch_norm(
        conv, center=True, scale=True, is_training=train, scope=scope)
    output = tf.nn.relu(conv_normed, name=scope.name)
    return output


def _activation_summary(x):
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    if name == 'conv1_1':
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, initializer=initializer,
                                  trainable=False)
    else:
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, wd):
    if (FLAGS.first_layer_ternary == False) & (name == 'conv1_1'):
        initializer = np.load(FLAGS.hot_start_dir + '/W_' + name + '.npy')
    else:
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(inputs, weights, train=False):
    # fc
    with tf.variable_scope('fc1') as scope:
        dim = inputs.get_shape()[1].value
        fc1 = reparametrization(
            inputs, [dim, 4], scope, weights[0], conv=False, train=train)
        if train:
            fc1 = tf.nn.dropout(fc1, FLAGS.dropout)
        _activation_summary(fc1)

    with tf.variable_scope('fc2') as scope:
        dim = fc1.get_shape()[1].value
        fc2 = reparametrization(
            fc1, [dim, 4], scope, weights[1], conv=False, train=train)
        if train:
            fc2 = tf.nn.dropout(fc2, FLAGS.dropout)
        _activation_summary(fc2)

    with tf.variable_scope('fc3') as scope:
        dim = fc2.get_shape()[1].value
        fc3 = reparametrization(
            fc2, [dim, 3], scope, weights[2], conv=False, train=train,
            activation=False)
        if train:
            fc3 = tf.nn.dropout(fc3, FLAGS.dropout)
        _activation_summary(fc3)

    # # Linear classifier
    # with tf.variable_scope('softmax_linear'):
    #     weights = _variable_with_weight_decay('weights',
    #                                           [fc.get_shape()[1], NUM_CLASSES],
    #                                           wd=FLAGS.wd_weights)
    #     biases = _variable_on_cpu('biases', [NUM_CLASSES],
    #                               tf.constant_initializer(0.0))
    #     softmax_linear = tf.add(tf.matmul(fc, weights), biases)
    #     _activation_summary(softmax_linear)

    return fc3


def draw_weights(sess):
    if WEIGHT_TYPE == 'BINARY':
        with tf.variable_scope('fc1', reuse=True) as scope:
            a = get_probs()
            a_ = sess.run(a)
            W = draw_ternary_weight(a_)
            np.save(FLAGS.train_dir + '/weights/W_' + scope.name + '.npy', W)

        with tf.variable_scope('fc2', reuse=True) as scope:
            a = get_probs()
            a_ = sess.run(a)
            W = draw_ternary_weight(a_)
            np.save(FLAGS.train_dir + '/weights/W_' + scope.name + '.npy', W)

        with tf.variable_scope('fc3', reuse=True) as scope:
            a = get_probs()
            a_ = sess.run(a)
            W = draw_ternary_weight(a_)
            np.save(FLAGS.train_dir + '/weights/W_' + scope.name + '.npy', W)
    elif WEIGHT_TYPE == 'TERNARY':
        with tf.variable_scope('fc1', reuse=True) as scope:
            a, b = get_probs()
            a_, b_ = sess.run([a, b])
            W = draw_ternary_weight(a_, b_)
            np.save(FLAGS.train_dir + '/weights/W_' + scope.name + '.npy', W)

        with tf.variable_scope('fc2', reuse=True) as scope:
            a, b = get_probs()
            a_, b_ = sess.run([a, b])
            W = draw_ternary_weight(a_, b_)
            np.save(FLAGS.train_dir + '/weights/W_' + scope.name + '.npy', W)

        with tf.variable_scope('fc3', reuse=True) as scope:
            a, b = get_probs()
            a_, b_ = sess.run([a, b])
            W = draw_ternary_weight(a_, b_)
            np.save(FLAGS.train_dir + '/weights/W_' + scope.name + '.npy', W)


def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

