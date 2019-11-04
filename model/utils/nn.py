"""
  hat.model.utils.nn

  Neural Network Redefine Layers
"""

# pylint: disable=unused-argument
# pylint: disable=unused-variable
# pylint: disable=redefined-outer-name
# pylint: disable=redefined-builtin


# __all__ = [
  
# ]


import tensorflow as tf

from hat.utils.counter import Counter


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
  def _inner(*args, **kwargs):
    layer = func(*args, **kwargs)
    # proc
    if 'block' in kwargs:
      block = kwargs['block']
      # ensure block is not None
      if block is not None:
        layer = block(layer)
    return layer
  return _inner


def get_name(name, tag=None, block:Block=None):
  """
    Get Layer Name
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
  def _func(x):
    for i in range(times):
      x = layer(*args, **kwargs)(x)
    return x
  return _func

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
    name = get_name('reshape', block=block)
  return tf.keras.layers.Reshape(
    target_shape=target_shape,
    name=name,
    **kwargs
  )

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
    name = get_name('flatten', block=block)
  return tf.keras.layers.Flatten(
    data_format=data_format,
    name=name,
    **kwargs
  )

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
    name = get_name('add', block=block)
  return tf.keras.layers.Add(
    name=name,
    **kwargs
  )

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
    name = get_name('concatenate', block=block)
  return tf.keras.layers.Concatenate(
    axis=axis,
    name=name,
    **kwargs
  )

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
    name = get_name('softmax', '', block=block) # 'Softmax'
  elif name is None:
    name = get_name('dense', block=block)
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
    **kwargs
  )

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
    name = get_name('dropout', block=block)
  return tf.keras.layers.Dropout(
    rate=rate,
    noise_shape=noise_shape,
    seed=seed,
    name=name,
    **kwargs
  )

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
    name = get_name('Maxpool2D', block=block)
  return tf.keras.layers.MaxPool2D(
    pool_size=pool_size,
    strides=strides,
    padding=padding,
    data_format=data_format,
    name=name,
    **kwargs
  )

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
    name = get_name('Avgpool2D', block=block)
  return tf.keras.layers.AvgPool2D(
    pool_size=pool_size,
    strides=strides,
    padding=padding,
    data_format=data_format,
    name=name,
    **kwargs
  )

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
    name = get_name('GlobalMaxpool2D', block=block)
  return tf.keras.layers.GlobalMaxPool2D(
    data_format=data_format,
    name=name,
    **kwargs
  )

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
    name = get_name('GlobalAvgpool2D', block=block)
  return tf.keras.layers.GlobalAvgPool2D(
    data_format=data_format,
    name=name,
    **kwargs
  )

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
    name = get_name('Conv2D', block=block)
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
    **kwargs
  )

@hat_nn
def relu(
  max_value=6,
  negative_slope=0,
  threshold=0,
  name=None,
  block:Block=None,
  **kwargs
  ):
  """
    RuLU Layer

    default:
      max_value: 6
  """
  if name is None:
    name = get_name('relu', block=block)
  return tf.keras.layers.ReLU(
    max_value=max_value,
    negative_slope=negative_slope,
    threshold=threshold,
    name=name,
    **kwargs
  )

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
    name = get_name('activation', block=block)
  return tf.keras.layers.Activation(
    activation=activation,
    name=name,
    **kwargs
  )

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
    name = get_name('BatchNormalization', block=block)
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
    **kwargs
  )


# Alias
input = tf.keras.layers.Input
model = tf.keras.models.Model
concat = concatenate
maxpool = maxpool2d
avgpool = avgpool2d
gmpool = globalmaxpool2d
gapool = globalavgpool2d
conv = conv2d
bn = batchnormalization


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
  _layers = model.layers
  _temp = []
  for l in _layers:
    if b2.name in l.name:
      _temp.append(l)
  print(_layers)
  print(_temp)
  # Successed

