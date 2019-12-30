# -*- coding: utf-8 -*-
"""Group

  File: 
    /hat/model/custom/layer/squeeze

  Description: 
    Group Convolution Layers
"""


import tensorflow as tf

from hat.model.custom.util import normalize_tuple
from hat.model.custom.util import conv_output_length
from hat.model.custom.layers import basic


# import setting
__all__ = [
    'GroupConv2D',]


class GroupConv2D(tf.keras.layers.Layer):
  """GroupConv2D
  
    Description:
      None
    
    Args:
      group: Int. 分组数
      filters: Int, default None. 分组卷积时的卷积核个数，
          默认保持
      kernel_size: list/tuple of Int or Int, default (1, 1). 
          卷积核尺寸
      strides: list/tuple of Int or Int, default (1, 1).
          卷积步长
      padding: Str in ('valid', 'same', 'full'), default 'valid'.
          填充方式
      data_format: Str, default None. `channels_last`(None)
          或`channels_first`.
      activation: Str or Activation function, default None.
          激活函数
      use_bias: Bool, default True. 是否使用偏置
      use_group_bias: Bool, default False. 是否使用组偏置
      kernel_* & bias_*: 权重/偏置的初始化器、正则化器等

    Returns:
      tf.Tensor

    Raises:
      None

    Usage:
      None
  """
  def __init__(
      self,
      group,
      filters=None,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='valid',
      data_format=None,
      activation=None,
      use_bias=True,
      use_group_bias=False,
      kernel_initializer='glorot_uniform',
      kernel_regularizer=None,
      kernel_constraint=None,
      bias_initializer='zeros',
      bias_regularizer=None,
      bias_constraint=None,
      trainable=True,
      **kwargs):
    super().__init__(trainable=trainable, **kwargs)
    self.group = group
    self.filters = filters
    self.kernel_size = normalize_tuple(kernel_size, 2)
    self.strides = normalize_tuple(strides, 2)
    self.padding = padding
    self.data_format = data_format or tf.keras.backend.image_data_format()
    self.activation = activation
    self.use_bias = use_bias
    self.use_group_bias = use_group_bias
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

  def build(self, input_shape):
    def output_length2d(inputs):
      outputs = []
      for inx, dim in enumerate(inputs):
        outputs.append(conv_output_length(
            dim,
            self.kernel_size[inx],
            self.padding,
            self.strides[inx]))
      return outputs
    
    input_shape = input_shape.as_list()
    channel_axis = self.data_format == 'channels_first' and 1 or -1
    channel = input_shape[channel_axis]
    assert channel % self.group == 0, f'[Error] channel cannot be ' \
        f'disencated by group. {channel % self.group}'
    self.middle_channel = channel // self.group
    self.output_channel_i = self.filters or self.middle_channel
    self.output_channel = self.output_channel_i * self.group
    if self.data_format == 'channels_first':
      middle_shape = (self.middle_channel, input_shape[2], 
                      input_shape[3], self.group)
      output_shape = (self.output_channel,
                      *output_length2d(input_shape[2:4]))
    else:
      middle_shape = (input_shape[1], input_shape[2], 
                      self.group, self.middle_channel)
      output_shape = (*output_length2d(input_shape[1:3]), 
                      self.output_channel)

    self.reshape_layer_1 = tf.keras.layers.Reshape(middle_shape)
    self.reshape_layer_2 = tf.keras.layers.Reshape(output_shape)
    self.kernel_layer = tf.keras.layers.Conv3D(
        filters=self.output_channel_i,
        kernel_size=self.kernel_size + (1,),
        strides=self.strides + (1,),
        padding=self.padding,
        data_format=self.data_format,
        use_bias=self.use_group_bias,
        kernel_initializer=self.kernel_initializer,
        kernel_regularizer=self.kernel_regularizer,
        kernel_constraint=self.kernel_constraint,
        bias_initializer=self.bias_initializer,
        bias_regularizer=self.bias_regularizer,
        bias_constraint=self.bias_constraint,)
    if self.use_bias:
      self.bias = basic.AddBias(
          2,
          data_format=self.data_format,
          bias_initializer=self.bias_initializer,
          bias_regularizer=self.bias_regularizer,
          bias_constraint=self.bias_constraint,)
    self.built = True
  
  def call(self, inputs, **kwargs):
    x = inputs
    x = self.reshape_layer_1(x)
    x = self.kernel_layer(x)
    x = self.reshape_layer_2(x)
    if self.use_bias:
      x = self.bias(x)
    if self.activation is not None:
      x = tf.keras.layers.Activation(self.activation)(x)
    return x

  def get_config(self):
    config = {
        'group': self.group,
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'activation': self.activation,
        'use_bias': self.use_bias,
        'use_group_bias': self.use_group_bias,
        'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
        'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
        'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
        'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
        'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
        'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),}
    return dict(list(super().get_config().items()) + list(config.items()))


# test part
if __name__ == "__main__":
  x_ = tf.keras.backend.placeholder((None, 8, 8, 16))
  print(GroupConv2D(4, kernel_size=3, name='Group_Conv_2D')(x_))
