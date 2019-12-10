# -*- coding: utf-8 -*-
"""Group

  File: 
    /hat/model/custom/layer/squeeze

  Description: 
    Group Convolution Layers
"""


import tensorflow as tf

from hat.model.custom.util import normalize_tuple


# import setting
__all__ = [
  'GroupConv2D',
]


class GroupConv2D(tf.keras.layers.Layer):
  """GroupConv2D
  
    Description:
      None
    
    Attributes:
      None

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
    
  def build(self, input_shape):
    # print(type(input_shape))
    # input_shape = int(input_shape)
    input_shape = input_shape.as_list()
    def output_length2d(inputs):
      assert self.padding in {'same', 'valid', 'full', 'causal'}
      dilated_size_1 = self.kernel_size[0]
      if padding in ['same', 'causal']:
        output_length = input_length
      elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
      elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
      return (output_length + stride - 1) // stride
      # from tensorflow.python.keras.utils.conv_utils import conv_output_length
      # return (conv_output_length(
      #   inputs[0],
      #   self.kernel_size[1],
      #   self.padding,
      #   self.strides[1],
      #   1
      # ),conv_output_length(
      #   inputs[1],
      #   self.kernel_size[1],
      #   self.padding,
      #   self.strides[1],
      #   1
      # ))
    channel_axis = self.data_format == 'channels_first' and 1 or -1
    # if self.data_format == 'channels_first':
    #   channel_axis = 1
    # else:
    #   channel_axis = -1
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
                      output_length(input_shape[2]), 
                      output_length(input_shape[3]))
    else:
      middle_shape = (input_shape[1], input_shape[2], 
                      self.group, self.middle_channel)
      output_shape = (*output_length(input_shape[1:3]), 
                      # output_length(), 
                      self.output_channel)

    self.reshape_layer_1 = tf.keras.layers.Reshape(middle_shape)
    self.reshape_layer_2 = tf.keras.layers.Reshape(output_shape)
    self.kernel_layer = tf.keras.layers.Conv3D(
      filters=self.output_channel_i,
      kernel_size=self.kernel_size + (1,),
      strides=self.strides + (1,),
      padding=self.padding,
      data_format=self.data_format,
      use_bias=self.use_bias,
    )

    self.built = True
  
  def call(self, inputs, **kwargs):
    x = inputs
    x = self.reshape_layer_1(x)
    x = self.kernel_layer(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    return self.reshape_layer_2(x)


# test part
if __name__ == "__main__":
  x_ = tf.keras.backend.placeholder((None, 8, 8, 16))
  print(GroupConv2D(4, padding='same')(x_))
