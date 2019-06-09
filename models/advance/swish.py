import tensorflow as tf
from tensorflow.python.keras.layers import Layer


class Swish(Layer):

  def __init__(self, **kwargs):
    super(Swish, self).__init__(**kwargs)
    self.supports_masking = True

  def call(self, inputs, **kwargs):
    return tf.nn.swish(inputs)
