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
                                            SeparableConv1D)
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops, nn, nn_ops


class GroupConv(Layer):
  """
    Group Convolution 2D
  """
  def __init__(self, groups, filters, kernel_size, strides=1, padding='valid',
               data_format=None, dilation_rate=1, activation=None,
               use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, kernel_constraint=None,
               bias_constraint=None, split=True, **kwargs):
    super().__init__(**kwargs)
    rank = 2
    self.groups = groups
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.split = split

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
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_dim})
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    # self._convolution_op = nn_ops.Convolution(
    #     input_shape,
    #     filter_shape=self.kernel.get_shape(),
    #     dilation_rate=self.dilation_rate,
    #     strides=self.strides,
    #     padding=op_padding.upper(),
    #     data_format=conv_utils.convert_data_format(self.data_format,
    #                                                self.rank + 2))
    self._convolution_op = K.conv3d
    self.built = True

  def call(self, inputs, **kwargs):
    outputs = self._convolution_op(inputs, self.kernel)

    if self.use_bias:
      if self.data_format == 'channels_first':
        # As of Mar 2017, direct addition is significantly slower than
        # bias_add when computing gradients. To use bias_add, we collapse Z
        # and Y into a single dimension to obtain a 4D input tensor.
        outputs_shape = outputs.shape.as_list()
        if outputs_shape[0] is None:
          outputs_shape[0] = -1
        outputs_4d = array_ops.reshape(outputs,
                                       [outputs_shape[0], outputs_shape[1],
                                        outputs_shape[2] * outputs_shape[3],
                                        outputs_shape[4]])
        outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
        outputs = array_ops.reshape(outputs_4d, outputs_shape)
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs