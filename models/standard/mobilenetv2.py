"""
  MobileNet-v2-50
  NOTE:
    模型默认输入尺寸是(None, 300, 300, 3)
    模型中间特征图的设计参照Googlenet-v4
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           2,042,770
    Trainable params:       2,011,330
    Non-trainable params:   31,440
"""

# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *
from hat.models.advance import AdvNet


class mobilenetv2(AdvNet):
  """
    MobileNet-v2
  """
  def args(self):
    self.IRB_TIME = [1, 2, 3, 4, 3, 2, 1]
    self.IRB_CONV = [16, 24, 32, 48, 64, 96, 160]
    self.IRB_EXPD = [1, 3, 3, 3, 3, 3, 3]
    self.IRB_STEP = [1, 2, 2, 1, 2, 2, 1]
    
    self.CONV = [32, 640]
    
  def build_model(self):

    _list = list(zip(
      self.IRB_TIME,
      self.IRB_CONV,
      self.IRB_EXPD,
      self.IRB_STEP,
    ))

    x_in = self.input(self.INPUT_SHAPE)

    x = self.conv_bn(x_in, self.CONV[0], 3, 2)

    for i in _list:
      x = self._inverted_residual_block(x, *i)

    x = self.conv_bn(x, self.CONV[1], 1, 1)
    x = self.GAPool(x)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')
    # x = self.reshape(x, (1, 1, self.CONV[1]))
    # x = self.conv(x, self.NUM_CLASSES, 1, activation='softmax', name='Softmax')  # 加速
    # x = self.flatten(x)

    return self.Model(inputs=x_in, outputs=x, name='mobilenetv2')

  def _bottleneck(self, x_in, filters, k, strides=None):

    if K.image_data_format() == 'channels_first':
      axis = 1
    else:
      axis = -1

    c = K.int_shape(x_in)[axis] * k

    x = self.conv(x_in, c, 1)
    x = self.bn(x)
    x = self.relu(x)
    x = self.dwconv(x, 3, strides or 1)
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x, filters, 1)
    x = self.bn(x)

    if strides:
      x_in = self.conv(x_in, filters, 3, strides)
      x_in = self.bn(x_in)

    x = self.add([x_in, x])
    x = self.relu(x)

    return x

  def _inverted_residual_block(self, x, n, filters, k, strides=1):

    x = self._bottleneck(x, filters, k, strides)

    x = self.repeat(self._bottleneck, n - 1, filters, k)(x)

    return x


# test part
if __name__ == "__main__":
  mod = mobilenetv2(DATAINFO={'INPUT_SHAPE': (224, 224, 3), 'NUM_CLASSES': 114}, built=True)
  mod.summary()
