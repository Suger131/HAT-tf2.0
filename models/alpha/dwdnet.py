"""
  DwDNet
  DepthwiseConv & Dense & SE
  NOTE:
    模型默认输入尺寸是(None, 300, 300, 3)
    模型中间特征图的设计参照Googlenet-v4
  NOTE:
    最新确定的输入尺寸是(None, 256, 256, 3)
    且Head部分有所改动
    默认为0.5M的参数量
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           510,154
    Trainable params:       506,986
    Non-trainable params:   3,168
"""

# pylint: disable=no-name-in-module

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import SGD
from hat.models.advance import AdvNet


class dwdnet(AdvNet):
  """
    DWD-Net-0.5M
  """
  def args(self):
    self.CONV_BIAS = False
    self.SE_T = 8
    self.THETA = 0.5
    self.DROP = 0.5
    self.HEAD_CONV = [24, 32, 48]
    self.CONV = [48, 96, 192]
    self.TIME = [3, 3, 3]

    self.OPT = SGD(lr=1e-3, momentum=.9, decay=5e-4)

  def build_model(self):
    self.axis = -1

    x_in = self.input(self.INPUT_SHAPE)
    
    # Head
    x = self._head(x_in)

    # Dense part
    x = self.repeat(self._dense, self.TIME[0], self.CONV[0])(x)
    x = self._transition(x)
    x = self.repeat(self._dense, self.TIME[1], self.CONV[1])(x)
    x = self._transition(x)
    x = self.repeat(self._dense, self.TIME[2], self.CONV[2])(x)
    # x = self._transition(x)

    # Output
    x = self.GAPool(x)
    # x = self.flatten(x)
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='dwdnet')

  def _head(self, x_in):
    """
      INPUT: 
        (224,224, 3)
      
      OUTPUT:
        (56, 56, 48)
    """
    # Conv with bn, to (256,256,a1)
    x = self.conv_bn(x_in, self.HEAD_CONV[0], 3, use_bias=self.CONV_BIAS)
    # Conv with bn, downsampling to (128,128,a2)
    x = self.conv_bn(x, self.HEAD_CONV[1], 3, 2, use_bias=self.CONV_BIAS)
    # DWConv, to (128,128,a2)
    x = self.dwconv(x, 3, use_bias=self.CONV_BIAS)
    # SE block
    x = self.SE(x, rate=self.SE_T)
    # Conv with bn, downsampling to (64,64,a3)
    x = self.conv_bn(x, self.HEAD_CONV[2], 3, 2, use_bias=self.CONV_BIAS)
    # DWConv, to (64,64,a3)
    x = self.dwconv(x, 3, use_bias=self.CONV_BIAS)
    # SE block
    x = self.SE(x, rate=self.SE_T)
    # DWConv, downsampling to (32,32,a3)
    x = self.dwconv(x, 3, 2, use_bias=self.CONV_BIAS)
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

    filters = int(K.int_shape(x_in)[self.axis] * self.THETA)

    x = self.conv(x_in, filters, 1, use_bias=self.CONV_BIAS)
    x = self.bn(x)
    x = self.relu(x)
    x = self.dwconv(x, 3, strides=strides, use_bias=self.CONV_BIAS)
    x = self.bn(x)

    return x


# test part
if __name__ == "__main__":
  mod = dwdnet(DATAINFO={'INPUT_SHAPE': (100, 100, 3), 'NUM_CLASSES': 114}, built=True)
  mod.summary()


# class dwdnet(AdvNet):
#   '''
#     DwDNet
#   '''
#   def args(self):
#     self.CONV_BIAS = False
#     self.CONV = [64, 128, 256, 256]
#     self.TIME = [3, 3, 3, 3]
#     self.SE_T = 4
#     self.THETA = 0.5
#     self.DROP = 0

#   def build_model(self):
#     self.axis = -1

#     x_in = self.input(self.INPUT_SHAPE)
    
#     # Head
#     x = self._head(x_in)

#     # Dense part
#     x = self.repeat(self._dense, self.TIME[0], self.CONV[0])(x)
#     x = self._transition(x)
#     x = self.repeat(self._dense, self.TIME[1], self.CONV[1])(x)
#     x = self._transition(x)
#     x = self.repeat(self._dense, self.TIME[2], self.CONV[2])(x)
#     x = self._transition(x, strides=1, crop=False)
#     x = self.repeat(self._dense, self.TIME[3], self.CONV[3])(x)

#     # Output
#     x = self.GAPool(x)
#     x = self.dropout(x, self.DROP)
#     x = self.local(x, self.NUM_CLASSES, activation='softmax')

#     return self.Model(inputs=x_in, outputs=x, name='dwdnet')

#   def _head(self, x_in):
#     """
#       INPUT: 
#         (300,300, 3)
      
#       OUTPUT:
#         (35, 35, 48)
#     """
#     # Extand the RGB to (300,300,24)
#     x = self.conv_bn(x_in, 64, 3, use_bias=self.CONV_BIAS)
#     # downsampling to (150,150,24)
#     x = self.dwconv(x, 3, 2, use_bias=self.CONV_BIAS)
#     # DWConv to (148,148,24)
#     x = self.dwconv(x, 3, padding='valid', use_bias=self.CONV_BIAS)
#     # SE block
#     x = self.SE(x, rate=self.SE_T)
#     # downsampling to (73,73,36)
#     x = self.dwconv(x, 3, 2, padding='valid', use_bias=self.CONV_BIAS)
#     # DWConv to (71,71,36)
#     x = self.dwconv(x, 3, padding='valid', use_bias=self.CONV_BIAS)
#     # SE block
#     x = self.SE(x, rate=self.SE_T)
#     # downsampling to (35,35,48)
#     x = self.dwconv(x, 3, 2, padding='valid', use_bias=self.CONV_BIAS)
#     # return result
#     return x

#   def _dense(self, x_in, filters):

#     x = self.conv(x_in, filters, 1, use_bias=self.CONV_BIAS)
#     x = self.bn(x)
#     x = self.relu(x)
#     x = self.dwconv(x, 3, padding='same', use_bias=self.CONV_BIAS)
#     x = self.SE(x, rate=self.SE_T)

#     x = self.concat([x_in, x])

#     return x

#   def _transition(self, x_in, strides=2, crop=True):

#     filters = int(K.int_shape(x_in)[-1] * self.THETA)
#     padding = 'valid' if crop else 'same'

#     x = self.conv(x_in, filters, 1, use_bias=self.CONV_BIAS)
#     x = self.bn(x)
#     x = self.relu(x)
#     x = self.dwconv(x, 3, strides=strides, padding=padding, use_bias=self.CONV_BIAS)
#     x = self.bn(x)

#     return x