import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils


class Swish(Layer):
  """
    Swish Layer
    
    Swish = x * Sigmoid(x)
  """
  def __init__(self, **kwargs):
    super(Swish, self).__init__(**kwargs)
    self.supports_masking = True

  def call(self, inputs, **kwargs):
    return tf.nn.swish(inputs)

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape
