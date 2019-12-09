# -*- coding: utf-8 -*-
"""Resolution

  File: 
    /hat/model/custom/layer/resolution

  Description: 
    Resolution transformation Layers
"""


import tensorflow as tf
# from tensorflow.python.framework import tensor_shape
# from tensorflow.keras import backend as Ks

from hat.model.custom.util import normalize_tuple


# import setting
__all__ = [
  'ResolutionScaling2D',
  'ResolutionScal2D',
]


class ResolutionScaling2D(tf.keras.layers.Layer):
  """ResolutionScaling2D
  
    Description:
      None
    
    Attributes:
      size: Int or list of 2 Int.

    Returns:
      tf.Tensor

    Raises:
      TypeError
      LenError

    Usage:
      None
  """
  def __init__(
      self,
      size: list,
      axis=-1,
      **kwargs):
    super().__init__(trainable=False, **kwargs)
    self.size = normalize_tuple(size, 2)
    self.axis = axis

  def call(self, inputs, **kwargs):
    x = inputs
    if self.axis == -1:
      _size = tf.keras.backend.int_shape(x)[1:3]
    else:
      _size = tf.keras.backend.int_shape(x)[2:4]
    
    dh = self.size[0] - _size[0]
    dw = self.size[1] - _size[1]
    nh = abs(dh) // 2
    nw = abs(dw) // 2
    lh = [nh, abs(dh) - nh]
    lw = [nw, abs(dw) - nw]
    
    if dh < 0:
      x = tf.keras.layers.Cropping2D([lh, [0, 0]])(x)
    elif dh > 0:
      x = tf.keras.layers.ZeroPadding2D([lh, [0, 0]])(x)
    if dw < 0:
      x = tf.keras.layers.Cropping2D([[0, 0], lw])(x)
    elif dw > 0:
      x = tf.keras.layers.ZeroPadding2D([[0, 0], lw])(x)

    return x

  # def compute_output_shape(self, input_shape):
  #   channel = input_shape[self.axis]
  #   if self.axis == -1:
  #     new_shape = [input_shape[0], self.size, channel]
  #   else:
  #     new_shape = [input_shape[0], channel, self.size]
  #   return tf.python.framework.tensor_shape.TensorShape(new_shape)

  def get_config(self):
    config = {
      'size': self.size,
      'axis': self.axis,
    }
    return dict(list(super().get_config().items()) + list(config.items()))


# Alias
ResolutionScal2D = ResolutionScaling2D


# test part
if __name__ == "__main__":
  x_ = tf.keras.backend.placeholder((None, 8, 8, 16))
  print(ResolutionScaling2D((6, 6))(x_))
  print(ResolutionScaling2D((10, 10))(x_))
  print(ResolutionScaling2D((10, 6))(x_))
  print(ResolutionScaling2D((10, 10)).\
        compute_output_shape((None, 6, 6, 1)))
  