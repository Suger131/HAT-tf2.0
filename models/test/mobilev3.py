"""
  MobileNet-v3-small
  
"""

# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import

from tensorflow.python.keras import backend as K
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


class HSigmoid(Layer):
  """
    Hard Sigmoid Layer
    
    Hard Sigmoid = Relu6(x+3)/6
  """
  def __init__(self, **kwargs):
    super(HSigmoid, self).__init__(**kwargs)
    self.supports_masking = True

  def call(self, inputs, **kwargs):
    return K.relu(inputs+3, max_value=6)/6


class HSE(Layer):
  """
    Hard SE
  """
  def __init__(self, rate=4, reg=None, name='', **kwargs):
    super().__init__(name=name)
    self.rate=rate
    self.reg=reg
  def build(self, input_shape):
    input_channels=int(input_shape[3])
    self.gap = GlobalAveragePooling2D()
    self.to11 = Reshape((1,1,input_channels))
    self.conv1=Conv2D(
      input_channels//self.rate,
      kernel_size=1,
      use_bias=False,
      kernel_regularizer=self.reg,
      name="Squeeze",)
    self.conv2=Conv2D(
      input_channels,
      1,
      use_bias=False,
      kernel_regularizer=self.reg,
      name="Excite",
    )
    super().build(input_shape)

  def call(self, inputs, **kwargs):
    x = self.gap(inputs)
    x = self.to11(x)
    x = self.conv1(x)
    x = ReLU(6)(x)
    x = self.conv2(x)
    x = HSigmoid()(x)
    return inputs * x

  def get_config(self):
    config = {
      'rate': self.rate,
      'reg': self.reg,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({
  'HSwish': HSwish,
  'HSigmoid': HSigmoid,
  'HSE': HSE,
})


class mobilev3s(AdvNet):
  """
    MobileNet-v3-small
  """
  def args(self):
    self.reg = l2(l=1e-5)
    self.D = 0.2
    self.OPT = SGD(lr=1e-4, momentum=0.9)

  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)
    x = x_in
    # First layer
    x = self.conv(x, 16, 3, 2, padding='same', use_bias=False, kernel_regularizer=self.reg)
    x = self.bn(x)
    x = HSwish()(x)
    
    # Neck Part
    x = self.bneck(x, 3, 16, 16, True, 'relu', 2)
    x = self.bneck(x, 3, 72, 24, False, 'relu', 2)
    x = self.bneck(x, 3, 88, 24, False, 'relu', 1)
    x = self.bneck(x, 5, 96, 40, True, 'hswish', 2)
    x = self.bneck(x, 5, 240, 40, True, 'hswish', 1)
    x = self.bneck(x, 5, 240, 40, True, 'hswish', 1)
    x = self.bneck(x, 5, 120, 48, True, 'hswish', 1)
    x = self.bneck(x, 5, 144, 48, True, 'hswish', 1)
    x = self.bneck(x, 5, 288, 96, True, 'hswish', 2)
    x = self.bneck(x, 5, 576, 96, True, 'hswish', 1)
    x = self.bneck(x, 5, 576, 96, True, 'hswish', 1)

    x = self.conv(x, 576, 1, use_bias=False, kernel_regularizer=self.reg)
    x = self.bn(x)
    x = HSwish()(x)
    x = self.GAPool(x)
    x = self.reshape(x, (1,1,576))
    x = self.conv(x, 1280, 1, kernel_regularizer=self.reg)
    x = self.bn(x)
    x = HSwish()(x)
    if self.D: x = self.dropout(x, self.D)
    x = self.conv(x, self.NUM_CLASSES, 1, activation='softmax', kernel_regularizer=self.reg, name='SoftMax')
    x = self.flatten(x)

    # x = self.local(x, self.NUM_CLASSES, activation='softmax')
    return self.Model(inputs=x_in, outputs=x, name='mobilenetv2')

  def bneck(self, x_in, k, exp, out, se, nl, s):
    channels = K.int_shape(x_in)[-1]
    x = x_in
    if nl=='relu':
      act=[self.relu,self.relu]
    elif nl=='hswish':
      act=[HSwish(),HSwish()]
    # Expand
    x = self.conv(x, exp, 1, padding='same', use_bias=False, kernel_regularizer=self.reg)
    x = self.bn(x)
    x = act[0](x)
    # DWConv
    # dw_padding=(k-1)//2
    # x = ZeroPadding2D(padding=dw_padding)(x)
    x = self.dwconv(x, k, s, padding='same', use_bias=False, depthwise_regularizer=self.reg)
    x = self.bn(x)
    if se:
      x = HSE(reg=self.reg)(x)
    x = act[1](x)
    # Output
    x = self.conv(x, out, 1, padding='same', use_bias=False, kernel_regularizer=self.reg)
    if s==1 and channels==out:
      x = self.add([x_in, x])
    return x

# test part
if __name__ == "__main__":
  mod = mobilev3s(DATAINFO={'INPUT_SHAPE': (224, 224, 3), 'NUM_CLASSES': 1000}, built=True)
  mod.summary()
  
  # from tensorflow.python.keras.utils import plot_model

  # plot_model(
  #   mod.model,
  #   to_file='mobile-v3-s.jpg',
  #   show_shapes=True)