import os
from pathlib import Path
import tensorflow as tf

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

tf.app.flags.DEFINE_string('train_dir',
                           os.path.expanduser('~/Downloads/LRnets/iris_train'),
                           "Directory where to write event logs and checkpoint.")
tf.app.flags.DEFINE_integer('epochs', 5000, "Number of epochs to run.")
tf.app.flags.DEFINE_integer('num_gpus', 1, "How many GPUs to use.")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            "Whether to log device placement.")
tf.app.flags.DEFINE_boolean('new_run', True,
                            "Whether this is a new run or not.")
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          "Starting learning rate.")
tf.app.flags.DEFINE_float('lr_decay_epochs', 500,
                          """Number of training epochs upon 
                          which learning rate is adjusted.""")

tf.app.flags.DEFINE_string('eval_dir',
                           os.path.join(str(Path.home()),
                                        'Downloads/LRnets/iris_eval'),
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 150,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")