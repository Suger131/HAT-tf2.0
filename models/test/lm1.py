"""
  Light Model v1
  
"""

# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import

import math
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from hat.models.advance import AdvNet
from tensorflow.python.keras.optimizers import SGD


class HSwish(Layer):
  """
    Hard Swish Layer
    
    Hard Swish = x*Relu6(x+3)/6
  """
  def __init__(self, **kwargs):
    super(HSwish, self).__init__(**kwargs)
    self.supports_masking = True

  def call(self, inputs, **kwargs):
    return inputs*K.relu(inputs+3, max_value=6)/6


class HConv2D(Layer):
  """
    Conv2D with BN
    Chain: 
      A: Conv-BN-Act
      B: BN-Act-Conv
  """
  def __init__(
    self, filters, kernel_size, strides=(1, 1), padding='same', activation='relu',
    use_bias=True, use_bn=True, use_dw=False, chain='A', reg=None, name='', **kwargs):
    super().__init__(name=name, **kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    if activation == 'hswish':
      self.activation = HSwish()
    else:
      self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.use_bn = use_bn
    self.use_dw = use_dw
    self.chain = chain
    self.reg=reg
  def build(self, input_shape):
    if not self.use_dw:
      self.conv = Conv2D(self.filters, self.kernel_size, self.strides, self.padding,
        use_bias=self.use_bias, kernel_regularizer=self.reg, name="Conv")
    else:
      self.conv = DepthwiseConv2D(self.kernel_size, self.strides, self.padding,
        use_bias=self.use_bias, depthwise_regularizer=self.reg, name="DWConv")
    if self.use_bn:
      self.bn = BatchNormalization()
    super().build(input_shape)

  def call(self, inputs, **kwargs):
    x = inputs
    if self.chain == 'A':
      x = self.conv(x)
      if self.use_bn: x = self.bn(x)
      if self.activation is not None: x = self.activation(x)
    elif self.chain == 'B':
      if self.use_bn: x = self.bn(x)
      if self.activation is not None: x = self.activation(x)
      x = self.conv(x)
    else:
      raise Exception("Chain must be A or B, not", self.chain)
    return x

  def get_config(self):
    if isinstance(self.activation, HSwish):
      activation = 'hswish'
    else:
      activation = activations.serialize(self.activation)
    config = {
      'filters': self.filters,
      'kernel_size': self.kernel_size,
      'strides': self.strides,
      'padding': self.padding,
      'activation': activation,
      'use_bias': self.use_bias,
      'use_bn': self.use_bn,
      'use_dw': self.use_dw,
      'chain': self.chain,
      'reg': self.reg,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({
  'HSwish': HSwish,
  'HConv2D': HConv2D,
})


class lm1(AdvNet):
  """
    LM v1
  """
  def args(self):
    self.resolution = 100
    self.reg = l2(l=1e-5)
    self.D = 0.2
    self.OPT = SGD(lr=1e-3, momentum=0.9)
  def build_model(self):
    
    x_in = self.input(shape=self.INPUT_SHAPE)
    x = x_in
    
    # Stem
    x = self.Stem()(x)
    # Block Part
    x = self.Block(3, 32, 16, 'relu', 2)(x)
    x = self.Block(3, 72, 24, 'relu', 2)(x)
    x = self.Block(3, 72, 24, 'relu', 1)(x)
    x = self.Block(5, 96, 32, 'hswish', 2)(x)
    x = self.Block(5, 96, 32, 'hswish', 1)(x)
    x = self.Block(5, 96, 32, 'hswish', 1)(x)
    x = self.Block(5, 96, 48, 'hswish', 1)(x)
    x = self.Block(5, 144, 48, 'hswish', 1)(x)
    x = self.Block(3, 288, 96, 'hswish', 2)(x)
    x = self.Block(3, 384, 96, 'hswish', 1)(x)
    x = self.Block(3, 512, 128, 'hswish', 1)(x)
    # Conv Output
    x = HConv2D(512, 1, activation=HSwish(), use_bias=False, reg=self.reg)(x)
    # Head
    x = self.Head()(x)
    
    return self.Model(inputs=x_in, outputs=x, name='lm1')

  def Stem(self):
    """Stem"""
    def Stem(x_in):
      x = x_in
      if self.resolution:
        x = self.proc_input_v2(self.resolution)(x)
      x = HConv2D(32, 3, 2, activation=HSwish(), use_bias=False, reg=self.reg)(x)
      return x
    return Stem

  def Head(self):
    """Head"""
    def Head(x_in):
      x = x_in
      x = self.GAPool(x)
      x = self.reshape(x, (1, 1, 512))
      x = HConv2D(1024, 1, activation=HSwish(), reg=self.reg)(x)
      if self.D:
        x = self.dropout(x, self.D)
      x = self.conv(x, self.NUM_CLASSES, 1, activation='softmax', kernel_regularizer=self.reg, name='SoftMax')
      x = self.flatten(x)
      return x
    return Head

  def Block(self, kernel_size, exp, out, nl, strides=1, use_bias=False):
    """Block"""
    def Block(x_in):
      channels = K.int_shape(x_in)[-1]
      x = x_in
      x = HConv2D(exp, kernel_size, 1, padding='same', use_bias=use_bias, activation=nl, reg=self.reg)(x)
      x = HConv2D(exp, kernel_size, strides, padding='same', use_bias=use_bias, activation=nl, reg=self.reg, use_dw=True)(x)
      x = HConv2D(out, 1, 1, padding='same', use_bias=use_bias, activation=None, reg=self.reg)(x)
      if strides == 1 and out == channels:
        x = self.add([x_in, x])
      return x
    return Block

  def proc_input_v2(self, resolution, kernel_size=7, padding='valid'):
    """ ConvT """
    def _func(x_in):
      x = x_in
      i1 = self.INPUT_SHAPE[0]
      if i1 >= resolution: return x
      s1 = math.ceil(resolution / i1)
      s2 = math.floor(resolution / i1)
      s = s1 if abs(resolution - i1 * s1 - kernel_size + s1) < abs(resolution - i1 * s2 - kernel_size + s2) else s2
      x = Conv2DTranspose(self.INPUT_SHAPE[-1], kernel_size, strides=s, padding=padding, name="ProcInputV2")(x)
      x = self.proc_input(x, resolution)
      return x
    return _func


# test part
if __name__ == "__main__":
  mod = lm1(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10}, built=True)
  mod.summary()
  
  # from tensorflow.python.keras.utils import plot_model

  # plot_model(
  #   mod.model,
  #   to_file='mobile-v3-s.jpg',
  #   show_shapes=True)