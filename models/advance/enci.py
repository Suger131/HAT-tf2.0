# pylint: disable=useless-super-delegation
# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import
# pylint: disable=attribute-defined-outside-init

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Initializer


class EfficientNetConvInitializer(Initializer):
  """Initialization for convolutional kernels.
  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas base_path we use a normal distribution. Similarly,
  tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
  a corrected standard deviation.

  # Arguments:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  # Returns:
    an initialization for the variable
  """
  def __init__(self):
    super(EfficientNetConvInitializer, self).__init__()

  def __call__(self, shape, dtype=None, partition_info=None):
    dtype = dtype or K.floatx()

    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)

