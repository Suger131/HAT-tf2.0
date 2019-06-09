# pylint: disable=no-member
# pylint: disable=attribute-defined-outside-init


import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import (activations, constraints, initializers,
                                     regularizers)
from tensorflow.python.keras.layers import (Conv1D, InputSpec, Layer,
                                            SeparableConv1D)
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops, nn, nn_ops


class GroupConv(Layer):
  """
    Group Conv nD(rank=n)

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
      rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
      groups: An integer, the number of the groups, e.g. "2" for 2 groups convolution.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
          kernel after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: A string, the name of the layer.
  """

  def __init__(self, groups, filters, kernel_size, strides=1, padding='valid',
               data_format=None, dilation_rate=1, activation=None,
               use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None,
               trainable=True, rank=2, split=True, name=None, **kwargs):
    super(GroupConv, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank
    self.split = split
    self.groups = groups
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    if (self.padding == 'causal' and not isinstance(self,
                                                    (Conv1D, SeparableConv1D))):
      raise ValueError('Causal padding is only supported for `Conv1D`'
                       'and ``SeparableConv1D`.')
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
    self.input_spec = InputSpec(ndim=self.rank + 2)

  def build(self, input_shape):
    input_shape_ = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      channel_axis = 1
      _in_list = [input_shape_[0]] + [input_shape_[channel_axis] // self.groups] + input_shape_[2:]
    else:
      channel_axis = -1
      _in_list = input_shape_[: self.rank + 1] + [input_shape_[self.rank + 1] // self.groups]
    if self.split:
      input_shape = tensor_shape.TensorShape(_in_list)
    else:
      input_shape = tensor_shape.TensorShape(input_shape)

    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    self.kernel = []
    for i in range(self.groups):
      self.kernel.append(self.add_weight(
          name=f'kernel_{i}',
          shape=kernel_shape,
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint,
          trainable=True,
          dtype=self.dtype))
    
    if self.use_bias:
      self.bias = []
      for i in range(self.groups):
        self.bias.append(self.add_weight(
            name=f'bias_{i}',
            shape=(self.filters,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype))
    else:
      self.bias = None
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_dim * self.groups if self.split else input_dim})
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape=self.kernel[0].get_shape(),
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=op_padding.upper(),
        data_format=conv_utils.convert_data_format(self.data_format,
                                                   self.rank + 2))
    self.built = True

  def call(self, inputs, **kwargs):
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1

    if self.rank == 1 and self.padding == 'causal':
      inputs = array_ops.pad(inputs, self._compute_causal_padding())
    
    if self.split:
      inputs = tf.split(inputs, axis=channel_axis, num_or_size_splits=[self.filters,] * self.groups)

    outputs = []
    for i in range(self.groups):
      if self.split:
        temp = self._convolution_op(inputs[i], self.kernel[i])
      else:
        temp = self._convolution_op(inputs, self.kernel[i])
      
      if self.use_bias:
        if self.data_format == 'channels_first':
          if self.rank == 1:
            # nn.bias_add does not accept a 1D input tensor.
            bias = array_ops.reshape(self.bias[i], (1, self.filters, 1))
            temp += bias
          if self.rank == 2:
            temp = nn.bias_add(temp, self.bias[i], data_format='NCHW')
          if self.rank == 3:
            # As of Mar 2017, direct addition is significantly slower than
            # bias_add when computing gradients. To use bias_add, we collapse Z
            # and Y into a single dimension to obtain a 4D input tensor.
            temp_shape = temp.shape.as_list()
            if temp_shape[0] is None:
              temp_shape[0] = -1
            temp_4d = array_ops.reshape(temp,
                                       [temp_shape[0], temp_shape[1],
                                        temp_shape[2] * temp_shape[3],
                                        temp_shape[4]])
            temp_4d = nn.bias_add(temp_4d, self.bias[i], data_format='NCHW')
            temp = array_ops.reshape(temp_4d, temp_shape)
        else:
          temp = nn.bias_add(temp, self.bias[i], data_format='NHWC')
      outputs.append(temp)

    outputs = tf.concat(outputs, axis=channel_axis)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.groups * self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.groups * self.filters] +
                                      new_space)

  def get_config(self):
    config = {
        'split': self.split,
        'groups' : self.groups,
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(GroupConv, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if self.data_format == 'channels_last':
      causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
    else:
      causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
    return causal_padding
