# pylint: disable=arguments-differ

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer


class DropConnect(Layer):

  def __init__(self, drop_connect_rate=0., **kwargs):
    super(DropConnect, self).__init__(**kwargs)
    self.drop_connect_rate = float(drop_connect_rate)

  def call(self, inputs, training=None, **kwargs):

    def drop_connect():
      keep_prob = 1.0 - self.drop_connect_rate

      # Compute drop_connect tensor
      batch_size = tf.shape(inputs)[0]
      random_tensor = keep_prob
      random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
      binary_tensor = tf.floor(random_tensor)
      output = (inputs / keep_prob) * binary_tensor
      return output

    return K.in_train_phase(drop_connect, inputs, training=training)

  def get_config(self):
    config = {
      'drop_connect_rate': self.drop_connect_rate,
    }
    base_config = super(DropConnect, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
