# -*- coding: utf-8 -*-
"""Squeeze

  File: 
    /hat/model/custom/layer/squeeze

  Description: 
    Squeeze transformation Layers
"""


import tensorflow as tf

from hat.model.custom.layers import swish


# import setting
__all__ = [
  'SqueezeExcitation',
]


class SqueezeExcitation(tf.keras.layers.Layer):
  """SqueezeExcitation
  
    Description:
      None
    
    Attributes:
      ratio: float in (0, n), default 0.25. 压缩比，超过1为拓展比
      filters: Int, optional. 如果使用filters，ratio会被忽略
      min_filters: Int, default 1. 限制最小的filters
      data_format: Str, default None. `channels_last`(None)
          或`channels_first`.

    Returns:
      tf.Tensor

    Raises:
      None

    Usage:
      None
  """
  def __init__(
      self,
      ratio=0.25,
      filters=None,
      min_filters=1,
      data_format=None,
      trainable=True,
      **kwargs):
    super().__init__(trainable=trainable, **kwargs)
    self.ratio = ratio
    self.filters = filters
    self.min_filters = min_filters
    self.data_format = data_format or tf.keras.backend.image_data_format()

  def build(self, input_shape):
    channel_axis = self.data_format == 'channel_first' and 1 or -1
    channel = input_shape[channel_axis]
    filters = self.filters is not None and int(self.filters) or \
              max(self.min_filters, int(channel * self.ratio))
    self.reshape_layer = tf.keras.layers.Reshape((1, 1, channel))
    self.kernel_layer_1 = tf.keras.layers.Conv2D(filters, 1, 1, 'same', use_bias=True)
    self.kernel_layer_2 = tf.keras.layers.Conv2D(channel, 1, 1, 'same', use_bias=True)
    self.built = True

  def call(self, inputs, **kwargs):
    # channel_axis = self.data_format == 'channel_first' and 1 or -1
    # channel = tf.keras.backend.int_shape(inputs)[channel_axis]
    # filters = self.filters is not None and int(self.filters) or \
    #           max(self.min_filters, int(channel * self.ratio))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    # x = tf.keras.layers.Reshape((1, 1, channel))(x)
    x = self.reshape_layer(x)
    # x = tf.keras.layers.Conv2D(filters, 1, 1, 'same', use_bias=True)(x)
    x = self.kernel_layer_1(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    # x = tf.keras.layers.Conv2D(channel, 1, 1, 'same', use_bias=True)(x)
    x = self.kernel_layer_2(x)
    # x = tf.keras.layers.Activation('sigmoid')(x)
    x = swish.HSigmoid()(x)
    return tf.keras.layers.Multiply()([inputs, x])

  def get_config(self):
    config = {
      'ratio': self.ratio,
      'min_filters': self.min_filters,
      'data_format': self.data_format,
    }
    return dict(list(super().get_config().items()) + list(config.items()))


# test part
if __name__ == "__main__":
  x_ = tf.keras.backend.placeholder((None, 8, 8, 16))
  print(SqueezeExcitation()(x_))
  
  
