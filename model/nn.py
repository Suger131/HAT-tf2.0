# -*- coding: utf-8 -*-
"""NN

  File: 
    /hat/model/nn

  Description: 
    Neural Network Redefine Layers
"""


import tensorflow as tf
from tensorflow.keras import optimizers

from hat.util import Counter
from hat.model.custom import layer


class Block(object):
  """
    Block
  """
  def __init__(self, 
      name=None,
      **kwargs):
    if name is None:
      name = f"Block{Counter('_block')}"
    self.name = name.capitalize()

  def __call__(self, inputs, **kwargs):
    return self.call(inputs, **kwargs)

  def call(self, inputs:tf.keras.layers.Layer, **kwargs):
    # print(inputs.name)
    return inputs


# Built-in Function


def hat_nn(func):
  """A `Decorator` of `hat.Network_v2.layers`

    What it does is to automatically apply `Block` processing to the layer that the function return.
    `Block` is `hat.model.utils.nn.Block`

    Usage:

    ```python
    @hat_nn
    def function(*args):
      return layer # Inherited from tf.keras.layers.Layer
    ```
  """
  def inner_layer(*args, **kwargs):
    # get output layer from function
    layer = func(*args, **kwargs)
    # check block and apply
    if 'block' in kwargs:
      block = kwargs['block']
      # ensure block is not None
      if block is not None:
        layer = block(layer)
    return layer
  return inner_layer


def get_hat_name(name, tag=None, block:Block=None):
  """A `Function` to get Layer name

    The function uses the `Counter`(hat.utils.counter.Counter) to count the layers.
    Then combine the number with the layer name.

    Return: 
      Str. The new name of the layer.
  """
  if tag is None:
    tag = name
  _name = name.capitalize()
  if tag != '':
    _name += f"_{Counter(tag)}"
  if block is not None:
    _name = f'{block.name}_{_name}'
  return _name


# Public Function


def get_layer_with_block(model:tf.keras.models.Model, block:Block):
  """A `Function` to get layers that belong to the block

    Return:
      List of `tf.keras.layers.Layer` or Empty List.
  """
  return [l for l in model.layers if block.name in l.name]


def get_layer_output_name(model):
  """获取需要中间输出的层的名字

    Description:
      None

    Args:
      model: Keras.Model. 模型

    Returns:
      List

    Raises:
      None
  """
  blacklist_words = [
      'flatten',
      'dropout',
      'activation',
      'relu',
      'add',
      'concatenate',
      'reshape']
  layers = [i.name for i in model.layers]
  new_layers = []
  for i in layers:
    bool_i = True
    for name in blacklist_words:
      if name in i.lower():
        bool_i = False
    if bool_i:
      new_layers.append(i)
  return new_layers


def get_layer_output_name_full(model):
  """获取所有中间输出的层的名字

    Description:
      None

    Args:
      model: Keras.Model. 模型

    Returns:
      List

    Raises:
      None
  """
  return [i.name for i in model.layers]


def get_layer_weight_name(model):
  """获取需要中间权重的层的名字

    Description:
      None

    Args:
      model: Keras.Model. 模型

    Returns:
      List

    Raises:
      None
  """
  layers = [i.name for i in model.layers]
  new_layers = []
  for name in layers:
    if model.get_layer(name=name).trainable_weights == []:
      continue
    new_layers.append(name)
  return new_layers


def get_layer_output(model, x, name):
  """获取中间输出

    Description:
      None

    Args:
      model: Keras.Model. 模型
      x: np.array. 训练数据
      name: Str. 层名

    Returns:
      np.array

    Raises:
      None
  """
  K = tf.keras.backend
  layer = K.function([model.input], [model.get_layer(name=name).output])
  return layer([x])[0]


def get_layer_weight(model, name):
  """获取中间权重

    Description:
      None

    Args:
      model: Keras.Model. 模型
      name: Str. 层名

    Returns:
      List of np.array

    Raises:
      None
  """
  return model.get_layer(name=name).get_weights()


def get_channels(Tensor):
  if tf.keras.backend.image_data_format() == 'channels_first':
    axis = 1
  else:
    axis = -1
  return tf.keras.backend.int_shape(Tensor)[axis]


def input(
    shape,
    batch_size=None,
    dtype=None,
    sparse=False,
    tensor=None,
    **kwargs):
  return tf.keras.layers.Input(
      shape=shape,
      batch_size=batch_size,
      dtype=dtype,
      sparse=sparse,
      tensor=tensor,
      name=f"Input",
      **kwargs)


def repeat(layer, times, *args, **kwargs):
  """`repeat` is a `Function` of building Repeat Layers

    # parm

      - layer: nn.layers(function)
      - time: int. Number of times that need to be repeated.
      - *args & **kwargs: parameters of the nn.layers(function).

    # Return: 
      
      a python function

    # Usage:
    ```python
      x = nn.repeat(nn.dense, 3, 128)(x)
    ```
  """
  def inner_layer(x):
    for i in range(times):
      x = layer(*args, **kwargs)(x)
    return x
  return inner_layer

@hat_nn
def reshape(
    target_shape, 
    name=None,
    block:Block=None,
    **kwargs):
  """
    Reshape Layer
  """
  if name is None:
    name = get_hat_name('reshape', block=block)
  return tf.keras.layers.Reshape(
      target_shape=target_shape,
      name=name,
      **kwargs)

@hat_nn
def flatten(
    data_format=None, 
    name=None,
    block:Block=None,
    **kwargs):
  """
    Flatten Layer
  """
  if name is None:
    name = get_hat_name('flatten', block=block)
  return tf.keras.layers.Flatten(
      data_format=data_format,
      name=name,
      **kwargs)

@hat_nn
def add(
    name=None,
    block:Block=None,
    **kwargs):
  """
    Add Layer

    input must be a list
  """
  if name is None:
    name = get_hat_name('add', block=block)
  return tf.keras.layers.Add(
      name=name,
      **kwargs)

@hat_nn
def concatenate(
    axis=-1,
    name=None,
    block:Block=None,
    **kwargs):
  """
    Concatenate Layer

    input must be a list
  """
  if name is None:
    name = get_hat_name('concatenate', block=block)
  return tf.keras.layers.Concatenate(
      axis=axis,
      name=name,
      **kwargs)

@hat_nn
def dense( 
    units, 
    activation='relu',
    use_bias=True, 
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', 
    kernel_regularizer=None, 
    bias_regularizer=None,
    activity_regularizer=None, 
    kernel_constraint=None, 
    bias_constraint=None, 
    name=None,
    block:Block=None,
    **kwargs):
  """
    Full Connect Layer
  """
  if activation == 'softmax':
    name = get_hat_name('softmax', '', block=block) # 'Softmax'
  elif name is None:
    name = get_hat_name('dense', block=block)
  return tf.keras.layers.Dense(
      units=units,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      name=name,
      **kwargs)

@hat_nn
def dropout(
    rate,
    noise_shape=None,
    seed=None,
    name=None,
    block:Block=None,
    **kwargs):
  """
    Dropout Layer
  """
  if name is None:
    name = get_hat_name('dropout', block=block)
  return tf.keras.layers.Dropout(
      rate=rate,
      noise_shape=noise_shape,
      seed=seed,
      name=name,
      **kwargs)

@hat_nn
def maxpool2d(
    pool_size=(2, 2), 
    strides=None, 
    padding='same',
    data_format=None,
    name=None,
    block:Block=None, 
    **kwargs):
  """
    Max Pooling 2D Layer
  """
  if name is None:
    name = get_hat_name('Maxpool2D', block=block)
  return tf.keras.layers.MaxPool2D(
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      name=name,
      **kwargs)

@hat_nn
def avgpool2d(
    pool_size=(2, 2), 
    strides=None, 
    padding='same',
    data_format=None,
    name=None,
    block:Block=None, 
    **kwargs):
  """
    Avg Pooling 2D Layer
  """
  if name is None:
    name = get_hat_name('Avgpool2D', block=block)
  return tf.keras.layers.AvgPool2D(
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      name=name,
      **kwargs)

@hat_nn
def globalmaxpool2d(
    data_format=None,
    name=None,
    block:Block=None, 
    **kwargs):
  """
    Global Max Pooling 2D Layer
  """
  if name is None:
    name = get_hat_name('GlobalMaxpool2D', block=block)
  return tf.keras.layers.GlobalMaxPool2D(
      data_format=data_format,
      name=name,
      **kwargs)

@hat_nn
def globalavgpool2d(
    data_format=None,
    name=None,
    block:Block=None, 
    **kwargs):
  """
    Global Avg Pooling 2D Layer
  """
  if name is None:
    name = get_hat_name('GlobalAvgpool2D', block=block)
  return tf.keras.layers.GlobalAvgPool2D(
      data_format=data_format,
      name=name,
      **kwargs)

@hat_nn
def conv2d(
    filters, 
    kernel_size, 
    strides=1, 
    padding='same',
    data_format=None, 
    dilation_rate=(1, 1), 
    activation=None, 
    use_bias=True,
    kernel_initializer='glorot_uniform', 
    bias_initializer='zeros',
    kernel_regularizer=None, 
    bias_regularizer=None, 
    activity_regularizer=None,
    kernel_constraint=None, 
    bias_constraint=None, 
    name=None,
    block:Block=None,
    **kwargs):
  """
    Conv2D Layer
  """
  if name is None:
    name = get_hat_name('Conv2D', block=block)
  return tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      name=name,
      **kwargs)

@hat_nn
def conv3d(
    filters, 
    kernel_size, 
    strides=1, 
    padding='same',
    data_format=None, 
    dilation_rate=(1, 1, 1), 
    activation=None, 
    use_bias=True,
    kernel_initializer='glorot_uniform', 
    bias_initializer='zeros',
    kernel_regularizer=None, 
    bias_regularizer=None, 
    activity_regularizer=None,
    kernel_constraint=None, 
    bias_constraint=None, 
    name=None,
    block:Block=None,
    **kwargs):
  """
    Conv2D Layer
  """
  if name is None:
    name = get_hat_name('Conv3D', block=block)
  return tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      name=name,
      **kwargs)

@hat_nn
def dwconv2d(
    kernel_size, 
    strides=1, 
    padding='same',
    depth_multiplier=1,
    data_format=None, 
    activation=None, 
    use_bias=True,
    depthwise_initializer='glorot_uniform', 
    bias_initializer='zeros',
    depthwise_regularizer=None, 
    bias_regularizer=None, 
    activity_regularizer=None,
    depthwise_constraint=None, 
    bias_constraint=None, 
    name=None,
    block:Block=None,
    **kwargs):
  """
    Conv2D Layer
  """
  if name is None:
    name = get_hat_name('DepthwiseConv2D', block=block)
  return tf.keras.layers.DepthwiseConv2D(
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      depth_multiplier=depth_multiplier,
      data_format=data_format,
      activation=activation,
      use_bias=use_bias,
      depthwise_initializer=depthwise_initializer,
      bias_initializer=bias_initializer,
      depthwise_regularizer=depthwise_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      depthwise_constraint=depthwise_constraint,
      bias_constraint=bias_constraint,
      name=name,
      # kernel_initializer=depthwise_initializer, # bug fix
      **kwargs)

@hat_nn
def relu(
    max_value=6,
    negative_slope=0,
    threshold=0,
    name=None,
    block:Block=None,
    **kwargs):
  """
    RuLU Layer

    default:
      max_value: 6
  """
  if name is None:
    name = get_hat_name('relu', block=block)
  return tf.keras.layers.ReLU(
      max_value=max_value,
      negative_slope=negative_slope,
      threshold=threshold,
      name=name,
      **kwargs)

@hat_nn
def activation(
    activation,
    name=None,
    block:Block=None,
    **kwargs):
  """
    Activation Layer
  """
  if name is None:
    name = get_hat_name('activation', block=block)
  return tf.keras.layers.Activation(
      activation=activation,
      name=name,
      **kwargs)

@hat_nn
def batchnormalization(
    axis=-1, 
    momentum=0.99, 
    epsilon=1e-3, 
    center=True, 
    scale=True,
    beta_initializer='zeros', 
    gamma_initializer='ones', 
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones', 
    beta_regularizer=None, 
    gamma_regularizer=None,
    beta_constraint=None, 
    gamma_constraint=None, 
    renorm=False, 
    renorm_clipping=None,
    renorm_momentum=0.99, 
    fused=None, 
    trainable=True, 
    virtual_batch_size=None,
    adjustment=None, 
    name=None,
    block:Block=None,
    **kwargs):
  """
    BatchNormalization Layer
  """
  if name is None:
    name = get_hat_name('BatchNormalization', block=block)
  return tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=center,
      scale=scale,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      moving_mean_initializer=moving_mean_initializer,
      moving_variance_initializer=moving_variance_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      beta_constraint=beta_constraint,
      gamma_constraint=gamma_constraint,
      renorm=renorm,
      renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum,
      fused=fused,
      trainable=trainable,
      virtual_batch_size=virtual_batch_size,
      adjustment=adjustment,
      name=name,
      **kwargs)

# =============
# custom layers
# =============

@hat_nn
def resolutionscal2d(
    size,
    data_format=None,
    name=None,
    block:Block=None,
    **kwargs):
  """ResolutionScaling2D"""
  if name is None:
    name = get_hat_name('resolutionscaling2d', block=block)
  return layer.ResolutionScal2D(
      size=size,
      data_format=data_format,
      name=name,
      **kwargs)

@hat_nn
def sqeuuezeexcitation(
    ratio=0.25,
    filters=None,
    min_filters=1,
    data_format=None,
    name=None,
    block:Block=None,
    **kwargs):
  """SqueezeExcitation"""
  if name is None:
    name = get_hat_name('se', block=block)
  return layer.SqueezeExcitation(
      ratio=ratio,
      filters=filters,
      min_filters=min_filters,
      data_format=data_format,
      name=name,
      **kwargs)

@hat_nn
def groupconv2d(
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
    name=None,
    block:Block=None,
    **kwargs):
  """GroupConv2D"""
  if name is None:
    name = get_hat_name('groupconv2d', block=block)
  return layer.GroupConv2D(
      group=group,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      activation=activation,
      use_bias=use_bias,
      use_group_bias=use_group_bias,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer,
      kernel_constraint=kernel_constraint,
      bias_initializer=bias_initializer,
      bias_regularizer=bias_regularizer,
      bias_constraint=bias_constraint,
      name=name,
      **kwargs)

@hat_nn
def addbias(
    rank,
    data_format=None,
    bias_initializer='zeros',
    bias_regularizer=None,
    bias_constraint=None,
    name=None,
    block:Block=None,
    **kwargs):
  """AddBias"""
  if name is None:
    name = get_hat_name('addbias', block=block)
  return layer.AddBias(
      rank=rank,
      data_format=data_format,
      bias_initializer=bias_initializer,
      bias_regularizer=bias_regularizer,
      bias_constraint=bias_constraint,
      name=name,
      **kwargs)


# Alias
model = tf.keras.models.Model
concat = concatenate
local = dense
maxpool = maxpool2d
avgpool = avgpool2d
gmpool = globalmaxpool2d
gapool = globalavgpool2d
conv = conv2d
dwconv = dwconv2d
bn = batchnormalization
se = sqeuuezeexcitation
gconv = groupconv2d


# test
if __name__ == "__main__":
  x_in = input((28,28,1))
  b1 = Block('Stem')
  b2 = Block()
  x = flatten(block=b1)(x_in)
  x = dense(128, block=b2)(x)
  x = dropout(0.5, block=b2)(x)
  x = dense(10, activation='softmax', block=b2)(x)
  model = model(x_in, x)
  model.summary()

  # try to get layers through Block.name
  # _layers = model.layers
  # _temp = []
  # for l in _layers:
  #   if b2.name in l.name:
  #     _temp.append(l)
  # print(_layers)
  # print(_temp)
  print(get_layer_with_block(model, b2))
  # Successed

