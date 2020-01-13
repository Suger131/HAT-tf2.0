# -*- coding: utf-8 -*-
"""Activation

  File: 
    /hat/model/custom/layer/activation

  Description: 
    Activation Layers
"""


import tensorflow as tf


class Swish(tf.keras.layers.Layer):
  """Swish
  
    Description:
      Swish = x * Sigmoid(x)
    
    Args:
      None

    Returns:
      tf.Tensor

    Raises:
      None

    Usage:
      None
  """
  def __init__(
      self,
      **kwargs):
    super().__init__(trainable=False, **kwargs)

  def call(self, inputs, **kwargs):
    return tf.nn.swish(inputs)


class HSigmoid(tf.keras.layers.Layer):
  """Hard Sigmoid
  
    Description:
      HSigmoid = x * Relu6(x + 3) / 6
    
    Args:
      None

    Returns:
      tf.Tensor

    Raises:
      None

    Usage:
      None
  """
  def __init__(
      self,
      **kwargs):
    super().__init__(trainable=False, **kwargs)

  def call(self, inputs, **kwargs):
    return tf.keras.backend.relu(inputs + 3, max_value=6) / 6


class HSwish(tf.keras.layers.Layer):
  """Hard Swish
  
    Description:
      HSwish = x * Relu6(x + 3) / 6
    
    Args:
      None

    Returns:
      tf.Tensor

    Raises:
      None

    Usage:
      None
  """
  def __init__(
      self,
      **kwargs):
    super().__init__(trainable=False, **kwargs)

  def call(self, inputs, **kwargs):
    return inputs * tf.keras.backend.relu(inputs + 3, max_value=6) / 6


# test part
if __name__ == "__main__":
  x_ = tf.keras.backend.placeholder((None, 8, 8, 16))
  print(HSigmoid()(x_))

