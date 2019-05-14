"""
  DwDNet
  DepthwiseConv & Dense & SE
  NOTE:
    模型默认输入尺寸是(None, 300, 300, 3)
    模型中间特征图的设计参照Googlenet-v4
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           2,042,770
    Trainable params:       2,011,330
    Non-trainable params:   31,440
"""


from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *


class dwdnet(NetWork, AdvNet):
  '''
    DwDNet
  '''
  def args(self):
    self.RGB_C = 48 // 6
    self.CONV_BIAS = False
    self.CONV = [48, 96, 192]
    self.TIME = [3, 3, 3]
    self.SE_T = 3
    self.THETA = 0.5
    self.DROP = 0

  def build_model(self):
    self.axis = -1

    x_in = self.input(self.INPUT_SHAPE)
    
    # Head
    x = self._head(x_in)

    # Dense part
    x = self.repeat(self._dense, self.TIME[0], x, self.CONV[0])
    x = self._transition(x)
    x = self.repeat(self._dense, self.TIME[1], x, self.CONV[1])
    x = self._transition(x)
    x = self.repeat(self._dense, self.TIME[2], x, self.CONV[2])

    # Output
    x = self.GAPool(x)
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='dwdnet')

  def _head(self, x_in):
    """
      INPUT: 
        (300,300, 3)
      
      OUTPUT:
        (35, 35, 48)
    """
    # Extand the RGB to (300,300,24)
    x = self.exrgb(x_in, self.RGB_C)
    # downsampling to (150,150,24)
    x = self.dwconv(x, 3, 2, use_bias=self.CONV_BIAS)
    # DWConv to (148,148,24)
    x = self.dwconv(x, 3, padding='valid', use_bias=self.CONV_BIAS)
    # SE block
    x = self.SE(x, rate=self.SE_T)
    # downsampling to (73,73,36)
    x = self.dwconv(x, 3, 2, padding='valid', use_bias=self.CONV_BIAS)
    # DWConv to (71,71,36)
    x = self.dwconv(x, 3, padding='valid', use_bias=self.CONV_BIAS)
    # SE block
    x = self.SE(x, rate=self.SE_T)
    # downsampling to (35,35,48)
    x = self.dwconv(x, 3, 2, padding='valid', use_bias=self.CONV_BIAS)
    # return result
    return x

  def _dense(self, x_in, filters):

    x = self.conv(x_in, filters, 1, use_bias=self.CONV_BIAS)
    x = self.bn(x)
    x = self.relu(x)
    x = self.dwconv(x, 3, padding='same', use_bias=self.CONV_BIAS)
    x = self.SE(x, rate=self.SE_T)

    x = self.concat([x_in, x])

    return x

  def _transition(self, x_in, strides=2, crop=True):

    filters = int(K.int_shape(x_in)[-1] * self.THETA)
    padding = 'valid' if crop else 'same'

    x = self.conv(x_in, filters, 1, use_bias=self.CONV_BIAS)
    x = self.bn(x)
    x = self.relu(x)
    x = self.dwconv(x, 3, strides=strides, padding=padding, use_bias=self.CONV_BIAS)
    x = self.bn(x)

    return x


# test part
if __name__ == "__main__":
  mod = dwdnet(DATAINFO={'INPUT_SHAPE': (300, 300, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
