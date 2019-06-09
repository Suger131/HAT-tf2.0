# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import
# pylint: disable=attribute-defined-outside-init
# pylint: disable=useless-super-delegation

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Initializer


class EfficientNetDenseInitializer(Initializer):
  """Initialization for dense kernels.
    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                      distribution='uniform').
    It is written out explicitly base_path for clarity.

    # Arguments:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    # Returns:
      an initialization for the variable
  """
  def __init__(self):
    super(EfficientNetDenseInitializer, self).__init__()

  def __call__(self, shape, dtype=None, partition_info=None):
    dtype = dtype or K.floatx()

    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)

