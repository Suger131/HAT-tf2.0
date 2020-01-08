"""
  DwDsNet
  DepthwiseConv & Dense & SE
  NOTE:
    默认输入尺寸是(None, 224, 224, 3)
  本模型默认总参数量[参考基准：ImageNet]：
    Total params:           510,154
    Trainable params:       506,986
    Non-trainable params:   3,168
"""

# pylint: disable=no-name-in-module

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import *
from hat.models.advance import AdvNet


class dwdsnet(AdvNet):
  """
    DwDsNet
  """

  def args(self):
    self.CONV_BIAS = False
    self.SE_T = 16
    self.THETA = 0.5
    self.DROP = 0.5
    self.HEAD_CONV = [24, 32, 48]
    self.CONV = [48, 48, 96, 192]
    self.TIME = [1, 3, 3, 3]

    self.OPT = Adam(lr=1e-4)
    # self.OPT = SGD(lr=1e-3, momentum=.9, decay=5e-4)

  def build_model(self):
    self.axis = -1

    x_in = self.input(self.INPUT_SHAPE)
    
    # Head
    x = self._head(x_in)

    # Dense part
    # Stage 3
    x = self.repeat(self._dense, self.TIME[0], self.CONV[0])(x)
    x = self._transition(x)
    # Stage 4
    x = self.repeat(self._dense, self.TIME[1], self.CONV[1])(x)
    x = self._transition(x)
    # Stage 5
    x = self.repeat(self._dense, self.TIME[2], self.CONV[2])(x)
    x = self._transition(x)
    # Stage 6
    x = self.repeat(self._dense, self.TIME[3], self.CONV[3])(x)

    # Output
    x = self.GAPool(x)
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='dwdsnet')

  def mixpool(self, x_in, pool_size=2, strides=2):
    x1 = self.maxpool(x_in, pool_size, strides)
    x2 = self.avgpool(x_in, pool_size, strides)
    x = self.add([x1, x2])
    return x

  def _head(self, x_in):
    """
      INPUT: 
        (224,224, 3)
      
      OUTPUT:
        (56, 56, 48)
    """
    # Stage 0
    x = x_in
    if self.INPUT_SHAPE[0] < 66:
      x = self.proc_input(x, 66)

    # Stage 1
    # path 1
    x1 = self.conv_bn(x, 32, 7, 2, use_bias=self.CONV_BIAS)
    # path 2
    x2 = self.conv(x, 32, 3, activation='relu', use_bias=self.CONV_BIAS)
    x2 = self.mixpool(x2, 3, 2)
    # merge
    x = self.concat([x1, x2])
    
    # Stage 2
    # path 1
    x1 = self.SE(x, rate=self.SE_T)
    x1 = self.conv_bn(x1, 24, 3, use_bias=self.CONV_BIAS)
    x1 = self.dwconv(x1, 3, 2, use_bias=self.CONV_BIAS)
    # path 2
    x2 = self.conv(x, 24, 3, activation='relu', use_bias=self.CONV_BIAS)
    x2 = self.mixpool(x2, 3, 2)
    # merge
    x = self.concat([x1, x2])
    
    # return result
    return x

  def _dense(self, x_in, filters):
    x = x_in
    x = self.SE(x, rate=self.SE_T)
    x = self.bn(x)
    x = self.conv(x, filters, 1, use_bias=self.CONV_BIAS)
    x = self.bn(x)
    x = self.relu(x)
    x = self.dwconv(x, 3, padding='same', use_bias=self.CONV_BIAS)
    x = self.concat([x_in, x])
    return x

  def _transition(self, x_in, strides=2):
    filters = int(K.int_shape(x_in)[self.axis] * self.THETA)
    x = x_in
    x = self.SE(x, rate=self.SE_T)
    x = self.bn(x)
    x = self.conv(x, filters, 1, use_bias=self.CONV_BIAS)
    x = self.bn(x)
    x = self.relu(x)
    x = self.dwconv(x, 3, strides=strides, use_bias=self.CONV_BIAS)
    return x

# test part
if __name__ == "__main__":
  mod = dwdsnet(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 100}, built=True)
  mod.summary()

