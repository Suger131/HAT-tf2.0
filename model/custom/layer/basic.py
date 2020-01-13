# -*- coding: utf-8 -*-
"""Squeeze

  File: 
    /hat/model/custom/layer/basic

  Description: 
    Basic Layers
"""


import tensorflow as tf


class AddBias(tf.keras.layers.Layer):
  """AddBias
  
    Description:
      None
    
    Args:
      rank: Int. 偏置的维数，比如"2"是2D的偏置。
      data_format: Str, default None. `channels_last`(None)
          或`channels_first`.
      bias_initializer: 偏置的初始器，如果为None，将使用默认值。 
      bias_regularizer: 偏置的正则化器。
      bias_constraint: Optional projection function to be applied 
          to the bias after being updated by an `Optimizer`.

    Returns:
      tf.Tensor

    Raises:
      None

    Usage:
      None
  """
  def __init__(
      self,
      rank,
      data_format=None,
      bias_initializer='zeros',
      bias_regularizer=None,
      bias_constraint=None,
      trainable=True,
      **kwargs):
    super().__init__(trainable=trainable, **kwargs)
    self.rank = rank
    self.data_format = data_format or tf.keras.backend.image_data_format()
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

  def build(self, input_shape):
    channel_axis = self.data_format == 'channel_first' and 1 or -1
    self.channel = input_shape[channel_axis]
    self.bias = self.add_weight(
        name='bias',
        shape=(self.channel,),
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True,
        dtype=self.dtype)
    self.built = True

  def call(self, inputs, **kwargs):
    if self.data_format == 'channels_first':
      if self.rank == 1:
        # nn.bias_add does not accept a 1D input tensor.
        outputs = inputs + tf.reshape(self.bias, (1, self.channel, 1))
      else:
        outputs = tf.nn.bias_add(inputs, self.bias, data_format='NCHW')
    else:
      outputs = tf.nn.bias_add(inputs, self.bias, data_format='NHWC')
    return outputs

  def get_config(self):
    config = {
        'rank': self.rank,
        'data_format': self.data_format,
        'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
        'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
        'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),}
    return dict(list(super().get_config().items()) + list(config.items()))


# test part
if __name__ == "__main__":
  x_ = tf.keras.backend.placeholder((None, 16))
  print(AddBias(1, data_format='channel_last')(x_))
