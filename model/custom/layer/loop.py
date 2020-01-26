# -*- coding: utf-8 -*-
"""Loop

  File: 
    /hat/model/custom/layer/loop

  Description: 
    Loop Layers
"""


import tensorflow as tf

from hat.core import log
from hat.model.util import normalize_tuple


class LoopDense(tf.keras.layers.Layer):
  """LoopDense

    Description:
      Loop version - Dense

    Args:
      loops: Int. Number of Loop.
      ratio: Float, default 0.1. Ratio of loop updates.
      *others: the same as keras.Dense

    Return:
      tf.Tensor
  """
  def __init__(
      self,
      loops: int,
      units: int,
      ratio=0.1,
      activation=None,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      **kwargs):
    super().__init__(
        activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
        **kwargs)
    self.loops = int(loops)
    self.units = int(units)
    self.ratio = float(ratio)
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

  def build(self, input_shape):
    input_shape = input_shape.as_list()
    channel = input_shape[-1]
    self.forward_layer = tf.keras.layers.Dense(
        units=self.units,
        activation=None,
        use_bias=self.use_bias,
        kernel_initializer=self.kernel_initializer,
        bias_initializer=self.bias_initializer,
        kernel_regularizer=self.kernel_regularizer,
        bias_regularizer=self.bias_regularizer,
        activity_regularizer=self.activity_regularizer,
        kernel_constraint=self.kernel_constraint,
        bias_constraint=self.bias_constraint,
        name='Forward')
    self.back_layer = tf.keras.layers.Dense(
        units=channel,
        activation=None,
        use_bias=self.use_bias,
        kernel_initializer=self.kernel_initializer,
        bias_initializer=self.bias_initializer,
        kernel_regularizer=self.kernel_regularizer,
        bias_regularizer=self.bias_regularizer,
        activity_regularizer=self.activity_regularizer,
        kernel_constraint=self.kernel_constraint,
        bias_constraint=self.bias_constraint,
        name='Back')
    self.built = True

  def call(self, inputs, **kwargs):
    del kwargs
    x = inputs
    for i in range(self.loops):
      xi = self.forward_layer(x)
      xi = self.activation(xi)
      xi = self.back_layer(xi)
      xi = self.activation(xi)
      xi = tf.keras.layers.Lambda(lambda x: x*self.ratio)(xi)
      x = tf.keras.layers.add([x, xi])
      # log.debug(f'{self.name} loop: {i + 1}', name=__name__)
    x = self.forward_layer(x)
    x = self.activation(x)
    return x

  def get_config(self):
    config = {
        'loops': self.loops,
        'units': self.units,
        'ratio': self.ratio,
        'activation': tf.keras.activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer':
            tf.keras.initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            tf.keras.initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            tf.keras.constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            tf.keras.constraints.serialize(self.bias_constraint)}
    return dict(list(super().get_config().items()) + list(config.items()))


class LoopConv2D(tf.keras.layers.Layer):
  """LoopConv2D

    Description:
      Loop version - Conv2D

    Args:
      loops: Int. Number of Loop.
      ratio: Float, default 0.1. Ratio of loop updates.
      *others: the same as keras.Conv2D

    Return:
      tf.Tensor
  """
  def __init__(
      self,
      loops: int,
      filters: int,
      kernel_size,
      strides=1,
      ratio=0.1,
      padding='valid',
      data_format=None,
      activation=None,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      **kwargs):
    super().__init__(
        activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
        **kwargs)
    self.loops = int(loops)
    self.filters = int(filters)
    self.ratio = float(ratio)
    self.kernel_size = normalize_tuple(kernel_size, 2)
    self.strides = normalize_tuple(strides, 2)
    self.padding = padding
    self.data_format = data_format or tf.keras.backend.image_data_format()
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

  def build(self, input_shape):
    input_shape = input_shape.as_list()
    channel_axis = 1 if self.data_format == 'channels_first' else -1
    channel = input_shape[channel_axis]
    self.forward_layer = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=self.kernel_size,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        activation=None,
        use_bias=self.use_bias,
        kernel_initializer=self.kernel_initializer,
        kernel_regularizer=self.kernel_regularizer,
        kernel_constraint=self.kernel_constraint,
        bias_initializer=self.bias_initializer,
        bias_regularizer=self.bias_regularizer,
        bias_constraint=self.bias_constraint,
        name='Forward')
    self.back_layer = tf.keras.layers.Conv2DTranspose(
        filters=channel,
        kernel_size=self.kernel_size,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        activation=None,
        use_bias=self.use_bias,
        kernel_initializer=self.kernel_initializer,
        kernel_regularizer=self.kernel_regularizer,
        kernel_constraint=self.kernel_constraint,
        bias_initializer=self.bias_initializer,
        bias_regularizer=self.bias_regularizer,
        bias_constraint=self.bias_constraint,
        name='Back')
    self.built = True
    
  def call(self, inputs, **kwargs):
    del kwargs
    x = inputs
    for i in range(self.loops):
      xi = self.forward_layer(x)
      xi = self.activation(xi)
      xi = self.back_layer(xi)
      xi = self.activation(xi)
      xi = tf.keras.layers.Lambda(lambda x: x*self.ratio)(xi)
      x = tf.keras.layers.add([x, xi])
      # log.debug(f'{self.name} loop: {i + 1}', name=__name__)
    x = self.forward_layer(x)
    x = self.activation(x)
    return x

  def get_config(self):
    config = {
        'loops': self.loops,
        'ratio': self.ratio,
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'activation': tf.keras.activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer':
            tf.keras.initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            tf.keras.initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            tf.keras.constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            tf.keras.constraints.serialize(self.bias_constraint)}
    return dict(list(super().get_config().items()) + list(config.items()))


# test part
if __name__ == "__main__":
  x_ = tf.keras.backend.placeholder((None, 256))
  print(LoopDense(3, 64)(x_))
  x2 = tf.keras.backend.placeholder((None, 8, 8, 16))
  print(LoopConv2D(3, 8, 3)(x2))
