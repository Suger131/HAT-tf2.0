# pylint: disable=no-member
# pylint: disable=attribute-defined-outside-init
# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import
# pylint: disable=useless-super-delegation
# pylint: disable=unused-variable

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints, initializers, regularizers
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops, nn


class GroupConv2D(Layer):
  """
    Group Convolution 2D

    Argument:
      groups: An int. The numbers of groups. Make sure [Channels] can be disdivided by groups.
      filters: An int. The numbers of groups filters. 0 means filters=channels//groups.(default 0)
      kernel_size: An int or tuple/list of 2 int, specifying the length of the convolution window.(default 1)
      others: The same as normal Convolution.
  """
  def __init__(self, groups:int, filters=0, kernel_size=1, strides=1, padding='valid',
               data_format=None, activation=None, use_bias=True, use_group_bias=False,
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None,
               bias_constraint=None, **kwargs):
    super().__init__(**kwargs)
    self.groups = groups
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides') + (1,)
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
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
    
    # Processing shape
    _shape = inputs.shape.as_list()
    if _shape[0] is None:
      _shape[0] = -1 # batch
    if self.data_format == 'channels_first':
      _data_format = 'NCHW'
      shape_i = [_shape[0], self.input_dim_i, _shape[2], _shape[3], self.groups]
    else:
      _data_format = 'NHWC'
      shape_i = [_shape[0], _shape[1], _shape[2], self.groups, self.input_dim_i]

    # Reshape to 5D
    outputs = array_ops.reshape(inputs, shape_i)
    
    # Convolution
    outputs = self._convolution_op(
      outputs,
      self.kernel,
      strides=self.strides,
      padding=self.padding,
      data_format=self.data_format,
    )

    # Group biases adding
    # NOTE: transpose the G & Ci, and add groups biases, then transpose back
    if self.use_group_bias:
      if self.data_format == 'channels_first':
        outputs = K.permute_dimensions(outputs, [0, 4, 2, 3, 1])
        t_shape = outputs.shape.as_list()
        if t_shape[0] is None:
          t_shape[0] = -1  # batch
        outputs = array_ops.reshape(
          outputs,
          [t_shape[0], t_shape[1], t_shape[2] * t_shape[3], t_shape[4]]
        )# reshape to 4D
        outputs = nn.bias_add(outputs, self.group_biases, data_format=_data_format)
        outputs = array_ops.reshape(outputs, t_shape)
        outputs = K.permute_dimensions(outputs, [0, 4, 2, 3, 1])
      else:
        outputs = K.permute_dimensions(outputs, [0, 1, 2, 4, 3])
        outputs = nn.bias_add(outputs, self.group_biases, data_format=_data_format)
        outputs = K.permute_dimensions(outputs, [0, 1, 2, 4, 3])

    # Processing shape (back)
    _shape = inputs.shape.as_list()
    if _shape[0] is None:
      _shape[0] = -1 # batch
    if self.data_format == 'channels_first':
      shape = [_shape[0], self.output_dim, _shape[2], _shape[3]]
    else:
      shape = [_shape[0], _shape[1], _shape[2], self.output_dim]
    
    # Reshape to 4D
    outputs = array_ops.reshape(outputs, shape)
    
    # Biases Adding
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.biases, data_format=_data_format)

    # Activation
    if self.activation is not None:
      outputs = self.activation(outputs)

    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()

    if not self.filters:
      return tensor_shape.TensorShape(input_shape)
    else:
      space = input_shape[2:] if self.data_format == 'channels_first' else input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
          space[i],
          self.kernel_size[i],
          padding=self.padding,
          stride=self.strides[i],
        )
      new_space.append(new_dim)

      if self.data_format == 'channels_first':
        return tensor_shape.TensorShape([input_shape[0], self.groups * self.filters] + new_space)
      else:
        return tensor_shape.TensorShape([input_shape[0]] + new_space + [self.groups * self.filters])

  def get_config(self):
    config = {
      'groups': self.groups,
      'filters': self.filters,
      'kernel_size': self.kernel_size,
      'strides': self.strides,
      'padding': self.padding,
      'data_format': self.data_format,
      'activation': activations.serialize(self.activation),
      'use_bias': self.use_bias,
      'use_group_bias': self.use_group_bias,
      'kernel_initializer': initializers.serialize(self.kernel_initializer),
      'bias_initializer': initializers.serialize(self.bias_initializer),
      'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
      'activity_regularizer': regularizers.serialize(self.activity_regularizer),
      'kernel_constraint': constraints.serialize(self.kernel_constraint),
      'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

# test
if __name__ == '__main__':
  x = K.placeholder((None, 8, 8, 16))
  print(GroupConv2D(4)(x))
  # print(GroupConv2D(4, kernel_size=3, activation='relu')(x))
  # print(GroupConv2D(4, kernel_size=3, strides=2, activation='relu')(x))
  # print(GroupConv2D(4, kernel_size=3, padding='same', activation='relu')(x))
  # print(GroupConv2D(4, 8, kernel_size=3, padding='same', activation='relu')(x))
  print(GroupConv2D(8, use_bias=False)(x))
  print(GroupConv2D(8, use_bias=False, use_group_bias=True)(x))
