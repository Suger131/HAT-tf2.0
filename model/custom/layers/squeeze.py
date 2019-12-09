# -*- coding: utf-8 -*-
"""Squeeze

  File: 
    /hat/model/custom/layer/squeeze

  Description: 
    Squeeze transformation Layers
"""


import tensorflow as tf


# import setting
__all__ = [
  'SqueezeExcitation',
]


class SqueezeExcitation(tf.keras.layers.Layer):
  """SqueezeExcitation
  
    Description:
      None
    
    Attributes:
      size: Int or list of 2 Int.

    Returns:
      tf.Tensor

    Raises:
      TypeError
      LenError

    Usage:
      None
  """
  def __init__(
      self,
      ratio=0.25,
      min_filters=1,
      data_format=None,
      **kwargs):
    self.ratio = ratio
    self.min_filters = min_filters
    self.data_format = tf.keras.backend.image_data_format()

    pass

  def call(self, inputs, **kwargs):
    if self.data_format == 'channel_first':
      channel_axis = 1
      spatial_dims = [2, 3]
    else:
      channel_axis = -1
      spatial_dims = [1, 2]
    channel = tf.keras.backend.int_shape(inputs)[channel_axis]
    filters = max(self.min_filters, int(channel * self.ratio))
    x = inputs
    x = tf.keras.layers.Lambda(lambda a: tf.keras.backend.mean(a, 
        axis=spatial_dims, keepdims=True))(x)
    x = tf.keras.layers.Conv2D(
      filters,
      kernel_size=1,
      strides=1,
      padding='same',
      use_bias=True)(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    x = tf.keras.layers.Conv2D(
      channel,
      kernel_size=1,
      strides=1,
      padding='same',
      use_bias=True)(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    return tf.keras.layers.Multiply()([inputs, x])




