"""
  再次封装的一些功能
  包含的类：
    AdvNet

"""


# pylint: disable=no-name-in-module
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.initializers import Initializer
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import get_custom_objects

from utils.counter import Counter


class ExtendRGB(Layer):
  """
    Extend the RGB channels

    Input:
      (batch, ..., 3)
    
    Output:
      (batch, ..., k*6)

    Usage:

    ```python
      x = ExtendRGB(4)(x) # got (batch, ..., 24)
    ```
  """

  def __init__(self, k, data_format=None, dilation_rate=1, trainable=False, **kwargs):
    super(ExtendRGB, self).__init__(trainable=trainable, **kwargs)
    self.k = k
    self.data_format = data_format
    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
    if self.data_format == 'channels_First':
      self.axis = 1
    else:
      self.axis = -1

  def build(self, input_shape):

    self.kernel = self._color_weight()

    self._convolution_op = K.conv2d
    self.built = True

  def call(self, x):
    if x.shape[self.axis] != 3:
      raise Exception(f"Input Tensor must have 3 channels(RGB), but got {x.shape[self.axis]}")
    x = self._convolution_op(x, self.kernel, padding='same',
          data_format=self.data_format, dilation_rate=self.dilation_rate)
    return x

  def _color_weight(self):
    _weight = []
    for i in range(3):
      i_ = i + 1 if i + 1 <= 2 else 0
      for j in range(self.k + 1):
        _t = [0, 0, 0]
        _t[i] = 1. / (1. + j / self.k)
        _t[i_] = j / self.k / (1. + j / self.k)
        _weight.append(_t)
      for j in range(1, self.k):
        _t = [0, 0, 0]
        _t[i_] = 1. / (1. + (self.k - j) / self.k)
        _t[i] = (self.k - j) / self.k / (1. + (self.k - j) / self.k)
        _weight.append(_t)
    _weight = K.variable(_weight)
    _weight = K.transpose(_weight)
    _weight = K.reshape(_weight, (1, 1, 3, 6 * self.k))
    return _weight

  def compute_output_shape(self, input_shape):
    input_shape[self.axis] = 6 * self.k
    return input_shape

  def get_config(self):
    config = {
        'k': self.k,
    }
    base_config = super(ExtendRGB, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


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

  def call(self, inputs):
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


class SqueezeExcitation(Layer):
  """
    SE-block (Squeeze & Excitation)

    This block would not change the shape of Tensor.

    Usage:
    
    ```python
      x = SqueezeExcitation()(x)
      # or
      x = SE()(x)
    ```
  """
  def __init__(self, input_filters=None, rate=16, activation='sigmoid', data_format=None,
               use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None, **kwargs):
    super(SqueezeExcitation, self).__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    
    self.input_filters = input_filters
    self.rate = rate
    self.data_format = data_format
    if self.data_format == 'channels_First':
      self.axis = 1
    else:
      self.axis = -1
    self.use_bias = use_bias
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

  def build(self, input_shape):

    channels = int(input_shape[self.axis])
    input_filters = self.input_filters or channels
    c = max((1, int(input_filters / self.rate)))

    self.kernel1 = self.add_weight(
        'kernel1',
        shape=[channels, c],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    self.kernel2 = self.add_weight(
        'kernel2',
        shape=[c, channels],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.biases1 = self.add_weight(
          'biases1',
          shape=(c,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
      self.biases2 = self.add_weight(
          'biases2',
          shape=(channels,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    
    self.built = True

  def call(self, inputs):
    # Global Average Pooling
    if self.data_format == 'channels_first':
      weights = K.mean(inputs, axis=[2, -1])
    else:
      weights = K.mean(inputs, axis=[1, -2])
    # FC 1
    weights = gen_math_ops.mat_mul(weights, self.kernel1)
    if self.use_bias:
      weights = nn.bias_add(weights, self.biases1)
    # FC 2
    weights = gen_math_ops.mat_mul(weights, self.kernel2)
    if self.use_bias:
      weights = nn.bias_add(weights, self.biases2)
    if self.activation is not None:
      weights = self.activation(weights)
    # reshape
    weights = Reshape((K.int_shape(weights)[1], 1, 1)
      if self.data_format == 'channels_first'
      else (1, 1, K.int_shape(weights)[-1]))(weights)
    # Scale
    outputs = tf.multiply(inputs, weights)
    return outputs

  def get_config(self):
    config = {
        'input_filters': self.input_filters,
        'rate': self.rate,
        'use_bias': self.use_bias,
        'activation': activations.serialize(self.activation),
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_initializer': self.bias_initializer,
        'bias_regularizer': self.bias_regularizer,
        'bias_constraint': self.bias_constraint,
    }
    base_config = super(SqueezeExcitation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Shuffle(_Merge):
  """
    Layer that shuffle and concatenate a list of inputs
  """
  def __init__(self, axis=-1, **kwargs):
    super(Shuffle, self).__init__(**kwargs)
    self.axis = axis
    self.supports_masking = True
    self._reshape_required = False

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # Used purely for shape validation.
    if not isinstance(input_shape, list) or len(input_shape) < 2:
      raise ValueError('A `Concatenate` layer should be called '
                       'on a list of at least 2 inputs')
    if all(shape is None for shape in input_shape):
      return
    reduced_inputs_shapes = [list(shape) for shape in input_shape]
    shape_set = set()
    for i in range(len(reduced_inputs_shapes)):
      del reduced_inputs_shapes[i][self.axis]
      shape_set.add(tuple(reduced_inputs_shapes[i]))
    if len(shape_set) > 1:
      raise ValueError('A `Concatenate` layer requires '
                       'inputs with matching shapes '
                       'except for the concat axis. '
                       'Got inputs shapes: %s' % (input_shape))

  def _merge_function(self, inputs):
    
    x = K.concatenate(inputs, axis=self.axis)
    
    _len = len(inputs)
    _shape = K.int_shape(x)[1:]
    _shapex = _shape[:-1] + (_len, _shape[-1] // _len)
    _transpose = list(range(len(_shapex)+1))
    _transpose = _transpose[:-2] + [_transpose[-1], _transpose[-2]]

    x = Reshape(_shapex)(x)
    x = tf.transpose(x, _transpose)
    x = Reshape(_shape)(x)

    return x 

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if not isinstance(input_shape, list):
      raise ValueError('A `Concatenate` layer should be called '
                       'on a list of inputs.')
    input_shapes = input_shape
    output_shape = list(input_shapes[0])
    for shape in input_shapes[1:]:
      if output_shape[self.axis] is None or shape[self.axis] is None:
        output_shape[self.axis] = None
        break
      output_shape[self.axis] += shape[self.axis]
    return tuple(output_shape)

  def compute_mask(self, inputs, mask=None):
    if mask is None:
      return None
    if not isinstance(mask, list):
      raise ValueError('`mask` should be a list.')
    if not isinstance(inputs, list):
      raise ValueError('`inputs` should be a list.')
    if len(mask) != len(inputs):
      raise ValueError('The lists `inputs` and `mask` '
                       'should have the same length.')
    if all(m is None for m in mask):
      return None
    # Make a list of masks while making sure
    # the dimensionality of each mask
    # is the same as the corresponding input.
    masks = []
    for input_i, mask_i in zip(inputs, mask):
      if mask_i is None:
        # Input is unmasked. Append all 1s to masks,
        masks.append(array_ops.ones_like(input_i, dtype='bool'))
      elif K.ndim(mask_i) < K.ndim(input_i):
        # Mask is smaller than the input, expand it
        masks.append(array_ops.expand_dims(mask_i, axis=-1))
      else:
        masks.append(mask_i)
    concatenated = K.concatenate(masks, axis=self.axis)
    return K.all(concatenated, axis=-1, keepdims=False)

  def get_config(self):
    config = {
        'axis': self.axis,
    }
    base_config = super(Shuffle, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Swish(Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        return tf.nn.swish(inputs)


class DropConnect(Layer):

    def __init__(self, drop_connect_rate=0., **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_connect_rate = float(drop_connect_rate)

    def call(self, inputs, training=None):

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


class EfficientNetConvInitializer(Initializer):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas base_path we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.

    # Arguments:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    # Returns:
      an initialization for the variable
    """
    def __init__(self):
        super(EfficientNetConvInitializer, self).__init__()

    def __call__(self, shape, dtype=None, partition_info=None):
        dtype = dtype or K.floatx()

        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.random_normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


class EfficientNetDenseInitializer(Initializer):
    """Initialization for dense kernels.
        This initialization is equal to
          tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                          distribution='uniform').
        It is written out explicitly base_path for clarity.

        # Arguments:
          shape: shape of variable
          dtype: dtype of variable
          partition_info: unused

        # Returns:
          an initialization for the variable
    """
    def __init__(self):
        super(EfficientNetDenseInitializer, self).__init__()

    def __call__(self, shape, dtype=None, partition_info=None):
        dtype = dtype or K.floatx()

        init_range = 1.0 / np.sqrt(shape[1])
        return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


# Short name
SE = SqueezeExcitation
ENCI = EfficientNetConvInitializer
ENDI = EfficientNetDenseInitializer


# envs
_CUSTOM_OBJECTS = {
  'ExtendRGB': ExtendRGB,
  'GroupConv': GroupConv,
  'SqueezeExcitation': SqueezeExcitation,
  'SE': SE,
  'Shuffle': Shuffle,
  'Swish': Swish,
  'DropConnect': DropConnect,
  'EfficientNetConvInitializer': EfficientNetConvInitializer,
  'EfficientNetDenseInitializer': EfficientNetDenseInitializer,
  'ENCI': ENCI,
  'ENDI': ENDI,
}
get_custom_objects().update(_CUSTOM_OBJECTS)


class AdvNet(object):
  
  def __init__(self, *args, **kwargs):
    return super().__init__(*args, **kwargs)

  def repeat(self, func, times, x, *args, **kwargs):
    for i in range(times):
      x = func(x, *args, **kwargs)
    return x

  def input(self, shape, batch_size=None, dtype=None, sparse=False, tensor=None,
            **kwargs):
    return Input(shape=shape, batch_size=batch_size, dtype=dtype, sparse=sparse,
                 tensor=tensor, name=f"Input", **kwargs)

  def reshape(self, x, target_shape, **kwargs):
    x = Reshape(target_shape=target_shape, name=f"Reshape_{Counter('reshape')}", **kwargs)(x)
    return x

  def add(self, x, **kwargs):
    """
      x must be a list
    """
    x = Add(name=f"Add_{Counter('add')}", **kwargs)(x)
    return x

  def concat(self, x, axis=-1, **kwargs):
    """
      x must be a list
    """
    x = Concatenate(axis=axis, name=f"Concat_{Counter('concat')}", **kwargs)(x)
    return x

  def local(self, x, units, activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
    name = f"Softmax" if activation=='softmax' else f"Local_{Counter('local')}"
    x = Dense(units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
              kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
              name=name, **kwargs)(x)
    return x

  def dropout(self, x, rate, noise_shape=None, seed=None, **kwargs):
    x = Dropout(rate=rate, noise_shape=noise_shape, seed=seed,
                name=f"Dropout_{Counter('dropout')}", **kwargs)(x)
    return x

  def flatten(self, x, data_format=None, **kwargs):
    x = Flatten(data_format=data_format, name=f"Faltten_{Counter('flatten')}",
                **kwargs)(x)
    return x

  def maxpool(self, x, pool_size=(2, 2), strides=None, padding='same',
              data_format=None, **kwargs):
    if not strides:
      strides = pool_size
    x = MaxPool2D(pool_size=pool_size, strides=strides, padding=padding,
                  data_format=data_format, name=f"MaxPool_{Counter('maxpool')}", **kwargs)(x)
    return x

  def avgpool(self, x, pool_size=(2, 2), strides=(2, 2), padding='same',
              data_format=None, **kwargs):
    x = AvgPool2D(pool_size=pool_size, strides=strides, padding=padding,
                  data_format=data_format, name=f"AvgPool_{Counter('avgpool')}", **kwargs)(x)
    return x

  def GAPool(self, x, data_format=None, **kwargs):
    x = GlobalAvgPool2D(data_format=data_format, name=f"GlobalAvgPool_{Counter('gapool')}", **kwargs)(x)
    return x

  def GMPool(self, x, data_format=None, **kwargs):
    x = GlobalMaxPool2D(data_format=data_format, name=f"GlobalMaxPool_{Counter('gmpool')}", **kwargs)(x)
    return x

  def conv(self, x, filters, kernel_size, strides=(1, 1), padding='same',
           data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None, name='', **kwargs):
    if type(kernel_size) == int:
      kernel_size = (kernel_size,) * 2
    if type(strides) == int:
      strides = (strides,) * 2
    if not name:
      name = f"Conv_{Counter('conv')}_F{filters}_K{'%sx%s' % kernel_size}_S{'%sx%s' % strides}"
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
               padding=padding, data_format=data_format, dilation_rate=dilation_rate,
               activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
               name=name, **kwargs)(x)
    return x

  def conv3d(self, x, filters, kernel_size, strides=(1, 1, 1), padding='same',
             data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True,
             kernel_initializer='glorot_uniform', bias_initializer='zeros',
             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
             kernel_constraint=None, bias_constraint=None, **kwargs):
    if type(kernel_size) == int:
      kernel_size = (kernel_size,) * 3
    if type(strides) == int:
      strides = (strides,) * 3
    x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides,
               padding=padding, data_format=data_format, dilation_rate=dilation_rate,
               activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
               name=f"Conv3D_{Counter('conv3d')}" +
                    f"_F{filters}" +
                    f"_K{'%sx%sx%s' % kernel_size}" +
                    f"_S{'%sx%sx%s' % strides}", **kwargs)(x)
    return x

  def dwconv(self, x, kernel_size, strides=(1, 1), padding='same', depth_multiplier=1,
             data_format=None, activation=None, use_bias=True,
             depthwise_initializer='glorot_uniform', bias_initializer='zeros',
             depthwise_regularizer=None, bias_regularizer=None,
             activity_regularizer=None, depthwise_constraint=None,
             bias_constraint=None, **kwargs):
    if type(kernel_size) == int:
      kernel_size = (kernel_size,) * 2
    if type(strides) == int:
      strides = (strides,) * 2
    name = f"DWConv_{Counter('dwconv')}_K{'%sx%s' % kernel_size}_S{'%sx%s' % strides}"
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding,
          depth_multiplier=depth_multiplier, data_format=data_format, activation=activation,
          use_bias=use_bias, depthwise_initializer=depthwise_initializer,
          bias_initializer=bias_initializer, depthwise_regularizer=depthwise_regularizer,
          bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
          depthwise_constraint=depthwise_constraint, bias_constraint=bias_constraint,
          name=name, **kwargs)(x)
    return x

  def groupconv(self, x, groups, filters, kernel_size, strides=(1, 1), padding='same',
           data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None, split=True, rank=2, **kwargs):
    if type(kernel_size) == int:
      kernel_size = (kernel_size,) * 2
    if type(strides) == int:
      strides = (strides,) * 2
    x = GroupConv(groups=groups, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=padding, data_format=data_format, dilation_rate=dilation_rate,
            activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, rank=rank,
            name=f"GroupConv_{Counter('conv')}" +
                 f"_G{groups}" +
                 f"_F{filters}" +
                 f"_K{'%sx%s' % kernel_size}", split=split, **kwargs)(x)
    return x

  def bn(self, x, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
         beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
         moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
         beta_constraint=None, gamma_constraint=None, renorm=False, renorm_clipping=None,
         renorm_momentum=0.99, fused=None, trainable=True, virtual_batch_size=None,
         adjustment=None, **kwargs):
    x = BatchNormalization(axis=axis, momentum=momentum, epsilon=epsilon, center=center,
                           scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                           moving_mean_initializer=moving_mean_initializer, moving_variance_initializer=moving_variance_initializer,
                           beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                           beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
                           renorm=renorm, renorm_clipping=renorm_clipping, renorm_momentum=renorm_momentum,
                           fused=fused, trainable=trainable, virtual_batch_size=virtual_batch_size,
                           adjustment=adjustment, name=f"BN_{Counter('bn')}", **kwargs)(x)
    return x

  def relu(self, x, **kwargs):
    x = Activation('relu', name=f"ReLU_{Counter('relu')}", **kwargs)(x)
    return x

  def activation(self, x, activation, **kwargs):
    x = Activation(activation=activation, name=f"{activation.capitalize()}_{Counter('relu')}",
                   **kwargs)(x)
    return x

  def conv_bn(self, x, filters, kernel_size, strides=(1, 1), padding='same',
           data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None, axis=-1, momentum=0.99, epsilon=1e-3,
           center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
           moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None,
           gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, renorm=False,
           renorm_clipping=None, renorm_momentum=0.99, fused=None, trainable=True,
           virtual_batch_size=None, adjustment=None, **kwargs):
    '''
      带有bn层的conv, 默认激活函数为ReLU
    '''
    x = self.conv(x, filters=filters, kernel_size=kernel_size, strides=strides,
                  padding=padding, data_format=data_format, dilation_rate=dilation_rate,
                  activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                  bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                  kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
    x = self.bn(x, axis=axis, momentum=momentum, epsilon=epsilon, center=center,
                scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                moving_mean_initializer=moving_mean_initializer, moving_variance_initializer=moving_variance_initializer,
                beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
                renorm=renorm, renorm_clipping=renorm_clipping, renorm_momentum=renorm_momentum,
                fused=fused, trainable=trainable, virtual_batch_size=virtual_batch_size, **kwargs)
    if activation == 'relu':
      x = self.relu(x, **kwargs)
    else:
      x = self.activation(x, activation, **kwargs)
    return x

  def exrgb(self, x, k, dilation_rate=1, data_format=None, **kwargs):
    """
      拓展RGB通道
    """
    x = ExtendRGB(k, dilation_rate=dilation_rate, data_format=data_format, 
                  name=f"Extend_RGB", **kwargs)(x)
    return x

  def SE(self, x, input_filters=None, rate=16, activation='sigmoid', data_format=None,
        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, **kwargs):
    x = SE(
      input_filters=input_filters, rate=rate, activation=activation, data_format=data_format,
      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint, name=f"SE_{Counter('se')}", **kwargs
    )(x)
    return x


if __name__ == "__main__":
  print(Counter('conv'))
  print(f"{Counter('conv')}")
