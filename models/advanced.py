"""
  再次封装的一些功能
  包含的类：
    AdvNet

"""


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
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import conv_utils

from utils.counter import Counter


@tf_export('keras.layers.ExtandRGB')
class ExtandRGB(Layer):
  """
    Extand the RGB channels
      
    Usage:
    ```python
      x = ExtandRGB()(x)
    ```

    Argument:\n
      x: \n
      4D Tensor, (None, rows, cols, 3) if channels_last(default)\n
      4D Tensor, (None, 3, rows, cols) if channels_first

    Return:\n
      x: \n
      4D Tensor, (None, rows, cols, 7) if channels_last(default)\n
      4D Tensor, (None, 7, rows, cols) if channels_first\n
      This 7 channels contains:\n
      0: Original R Channel\n
      1: Original G Channel\n
      2: Original B Channel\n
      3: R + G + B
      4: R + G\n
      5: R + B\n
      6: G + B\n
  """

  def __init__(self, axis=-1, data_format='channels_last', **kwargs):
    super(ExtandRGB, self).__init__(trainable=False, **kwargs)
    self.data_format = data_format
    self.axis = axis
    if self.data_format == 'channels_First':
      self.axis = 1
    else:  
      if self.axis != -1:
        print(f"[Warning] data format is channels last, axis must be -1, but got {self.axis}. So use -1")
        self.axis = -1
    self._gray = [0.299, 0.587, 0.114]
    super(ExtandRGB, self).__init__(**kwargs)

  def call(self, x):
    if x.shape[self.axis] != 3:
      raise Exception(f"Input Tensor must have 3 channels(RGB), but got {x.shape[self.axis]}")
    _x = tf.split(x, axis=self.axis, num_or_size_splits=[1, 1, 1])
    _y = [_x[0] * self._gray[0] + _x[1] * self._gray[1] + _x[2] * self._gray[2],
          (_x[0] + _x[1]) / 2,
          (_x[0] + _x[2]) / 2,
          (_x[1] + _x[2]) / 2]
    x = tf.concat([*_x, *_y], axis=self.axis)
    return x

  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_First':
      return (input_shape[0], 7, input_shape[2], input_shape[3])
    else:
      return (input_shape[0], input_shape[1], input_shape[2], 7)


@tf_export('keras.layers.GroupConv')
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


@tf_export('keras.layers.SqueezeExcitation', 'keras.layers.SE')
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
  def __init__(self, rate=16, activation='sigmoid', data_format=None, kernel_initializer='glorot_uniform',
               kernel_regularizer=None, activity_regularizer=None, kernel_constraint=None, **kwargs):
    super(SqueezeExcitation, self).__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    
    self.rate = rate
    self.data_format = data_format
    if self.data_format == 'channels_First':
      self.axis = 1
    else:
      self.axis = -1
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)

  def build(self, input_shape):
    channels = int(input_shape[self.axis])

    self.kernel1 = self.add_weight(
        'kernel1',
        shape=[channels, channels // self.rate],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    self.kernel2 = self.add_weight(
        'kernel2',
        shape=[channels // self.rate, channels],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    
    self.built = True

  def call(self, inputs):
    # Global Average Pooling
    if self.data_format == 'channels_first':
      weights = K.mean(inputs, axis=[2, -1])
    else:
      weights = K.mean(inputs, axis=[1, -2])
    # FC 1
    weights = gen_math_ops.mat_mul(weights, self.kernel1)
    # FC 2
    weights = gen_math_ops.mat_mul(weights, self.kernel2)
    if self.activation is not None:
      weights = self.activation(weights)
    # reshape
    weights = Reshape((1, 1, K.int_shape(weights)[-1]))(weights)
    # Scale
    outputs = tf.multiply(inputs, weights)
    return outputs

  def get_config(self):
    config = {
        'rate': self.rate,
        'activation': activations.serialize(self.activation),
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
    }
    base_config = super(SqueezeExcitation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


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

  def rgb_extand(self, x, axis=-1, data_format=None, **kwargs):
    """
      拓展RGB通道, 原本的3个通道拓展成7个通道\n
      详情可参考`ExtandRGB`类
    """
    x = ExtandRGB(axis=axis, data_format=data_format, 
                   name=f"RGB_Extand_{Counter('rgb_extand')}", **kwargs)(x)
    return x

  def SE(self, x, rate=16, activation='sigmoid', data_format=None, kernel_initializer='glorot_uniform',
         kernel_regularizer=None, activity_regularizer=None, kernel_constraint=None, **kwargs):
    x = SE(rate=rate, activation=activation, data_format=data_format, kernel_initializer=kernel_initializer,
           kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer,
           kernel_constraint=kernel_constraint, name=f"SE_{Counter('se')}")(x)
    return x


# Short name
SE = SqueezeExcitation


# envs
_CUSTOM_OBJECTS = {'ExtandRGB': ExtandRGB,
                   'GroupConv': GroupConv,
                   'SqueezeExcitation': SqueezeExcitation,
                   'SE': SE}


if __name__ == "__main__":
  print(Counter('conv'))
  print(f"{Counter('conv')}")
