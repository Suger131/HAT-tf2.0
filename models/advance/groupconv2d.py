# pylint: disable=no-member
# pylint: disable=attribute-defined-outside-init
# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import
# pylint: disable=useless-super-delegation
# pylint: disable=unused-variable

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import (activations, constraints, initializers,
                                     regularizers)
from tensorflow.python.keras.layers import (Conv1D, InputSpec, Layer,
                                            SeparableConv1D, Reshape)
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops, nn, nn_ops


class GroupConv(Layer):
  """
    Group Convolution 2D

    Argument:
      groups: An int. The numbers of groups. Make sure [Channels] can be disdivided by groups.
      filters: An int. The numbers of groups filters. 0 means filters=channels//groups.(default 0)
      kernel_size: An int or tuple/list of 2 int, specifying the length of the convolution window.(default 1)
      others: The same as normal Convolution.
  """
  def __init__(self, groups, filters=0, kernel_size=1, strides=1, padding='valid',
               data_format=None, dilation_rate=1, activation=None, use_bias=True, 
               use_group_bias=False, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, kernel_constraint=None,
               bias_constraint=None, **kwargs):
    super().__init__(**kwargs)
    # rank = 2
    self.groups = groups
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, 2, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides') + (1,)
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, 3, 'dilation_rate')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.use_group_bias = use_group_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    
    input_dim = int(input_shape[channel_axis])
    # print(input_shape, input_dim, self.groups)
    assert not input_dim % self.groups
    self.input_dim_i = input_dim // self.groups
    self.output_dim_i = self.filters or self.input_dim_i
    self.output_dim = self.output_dim_i * self.groups

    kernel_shape = self.kernel_size + (1, self.input_dim_i, self.output_dim_i)
    biases_shape = (self.output_dim,)
    group_biases_shape = (self.groups,)

    self.kernel = self.add_weight(
      name='kernel',
      shape=kernel_shape,
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      trainable=True,
      dtype=self.dtype
    )
    if self.use_bias:
      self.biases = self.add_weight(
        name='biases',
        shape=biases_shape,
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True,
        dtype=self.dtype
      )
    else:
      self.biases = None
    if self.use_group_bias:
      self.group_biases = self.add_weight(
        name='group_biases',
        shape=group_biases_shape,
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True,
        dtype=self.dtype
      )
    else:
      self.group_biases = None

    self._convolution_op = K.conv3d
    self.built = True

  def call(self, inputs, **kwargs):
    
    _shape = list(K.int_shape(inputs))
    if _shape[0] is None:
      _shape = _shape[1:]
    if self.data_format == 'channels_first':
      channel_axis = 1
      shape_i = [self.input_dim_i] + _shape[1:3] + [self.groups]
    else:
      channel_axis = -1
      shape_i = _shape[ :2] + [self.groups, self.input_dim_i]

    x = Reshape(shape_i)(inputs)
    
    x = self._convolution_op(
      x,
      self.kernel,
      strides=self.strides,
      padding=self.padding,
      data_format=self.data_format,
    )

    if self.use_group_bias:
      pass

    print(x.shape, self.kernel.shape, self.strides)
    _shape = list(K.int_shape(x))
    if _shape[0] is None:
      _shape = _shape[1:]
    if self.data_format == 'channels_first':
      shape = [self.output_dim] + _shape[1:3]
    else:
      shape = _shape[ :2] + [self.output_dim]
    x = Reshape(shape)(x)
    
    if self.use_bias:
      
      if self.data_format == 'channels_first':
        _data_format = 'NCHW'
      else:
        _data_format = 'NHWC'
      
      x = nn.bias_add(x, self.biases, data_format=_data_format)

    if self.activation is not None:
      x = self.activation(x)

    return x

# test
if __name__ == '__main__':
  x = K.placeholder((None, 8, 8, 16))
  print(GroupConv(4)(x))
  print(GroupConv(4, kernel_size=3, activation='relu')(x))
  print(GroupConv(4, kernel_size=3, strides=2, activation='relu')(x))
  print(GroupConv(4, kernel_size=3, padding='same', activation='relu')(x))
  print(GroupConv(4, 8, kernel_size=3, padding='same', activation='relu')(x))
