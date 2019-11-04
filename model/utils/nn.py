"""
  hat.model.utils.nn

  Neural Network Redefine Layers
"""


__all__ = [
  'nn'
]


import tensorflow as tf

from hat.utils.counter import Counter


class Block(object):
  """
    Block
  """
  def __init__(self, 
    name=None,
    *args, **kwargs):
    if name is None:
      name = f"Block{Counter('_block')}"
    self.name = name.capitalize()

  def __call__(self, inputs, **kwargs):
    return self.call(inputs, **kwargs)

  def call(self, inputs:tf.keras.layers.Layer, **kwargs):
    print(inputs.name)
    return inputs


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


# rename
Input = tf.keras.layers.Input
Model = tf.keras.models.Model


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
  layer = tf.keras.layers.Dense(
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
  return layer

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
  layer = tf.keras.layers.Dropout(
    rate=rate,
    noise_shape=noise_shape,
    seed=seed,
    name=name,
    **kwargs
  )
  return layer

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
  layer = tf.keras.layers.Flatten(
    data_format=data_format,
    name=name,
    **kwargs
  )
  return layer


# test
if __name__ == "__main__":
  x_in = Input((28,28,1))
  b1 = Block('Stem')
  b2 = Block()
  x = flatten(block=b1)(x_in)
  x = dense(128, block=b2)(x)
  x = dropout(0.5, block=b2)(x)
  x = dense(10, activation='softmax', block=b2)(x)
  model = Model(x_in, x)
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

