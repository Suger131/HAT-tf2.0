# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import
# pylint: disable=attribute-defined-outside-init

import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints, initializers, regularizers
from tensorflow.python.keras.layers import *
from tensorflow.python.ops import gen_math_ops, nn


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

  def call(self, inputs, **kwargs):
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
