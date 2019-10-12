"""
  CNN 模型
  本模型默认总参数量[参考基准：fruits]：
    Total params:           1,632,080
    Trainable params:       1,632,080
    Non-trainable params:   0
"""

# pylint: disable=no-name-in-module

import math
from tensorflow.python.keras.layers import Conv2DTranspose
from hat.models.advance import AdvNet
from tensorflow.python.keras.optimizers import SGD


# import setting
__all__ = [
  'cnn',
  'cnn32',
  'cnn64',
  'cnn128',
  'cnn256',
]


class cnn(AdvNet):
  """
    CNN
  """

  def __init__(self, conv, local, name='', **kwargs):
    self.CONV = conv
    self.LOCAL = local
    self.NAME = name
    super().__init__(**kwargs)

  def args(self):

    self.USE_MIN_SIZE = True
    self.MIN_SIZE = 100

    self.CONV = self.CONV
    self.LOCAL = self.LOCAL

    self.DROP = 0.5

    self.OPT = SGD(lr=0.1, momentum=.9, decay=4e-4)
    
  def build_model(self):

    x_in = self.input(self.INPUT_SHAPE)
    x = x_in

    # proc_input
    i1 = self.INPUT_SHAPE[0]
    if self.USE_MIN_SIZE and i1 < self.MIN_SIZE:
      s1 = math.ceil(self.MIN_SIZE / i1)
      s2 = math.floor(self.MIN_SIZE / i1)
      s = s1 if abs(self.MIN_SIZE - i1 * s1 - 7 + s1) < abs(self.MIN_SIZE - i1 * s2 - 7 + s2) else s2
      x = Conv2DTranspose(self.INPUT_SHAPE[-1], 7, strides=s, padding='valid')(x)
      x = self.proc_input(x, self.MIN_SIZE)

    # conv
    for i in self.CONV:
      x = self.conv(x, i, 5, activation='relu')
      x = self.maxpool(x)

    # local
    x = self.flatten(x)
    for i in self.LOCAL:
      x = self.local(x, i)
      x = self.dropout(x, self.DROP)

    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name=self.NAME)


def cnn32(**kwargs):
  """
    CNN-32

    Conv: [16, 32, 64, 128]
    
    Local: [512, 256]
  """
  return cnn(
    conv=[16, 32, 64, 128],
    local=[512, 256],
    name='cnn32',
    **kwargs
  )


def cnn64(**kwargs):
  """
    CNN-64

    Conv: [16, 32, 64, 128, 256]
    
    Local: [1024, 256]
  """
  return cnn(
    conv=[16, 32, 64, 128, 256],
    local=[1024, 256],
    name='cnn64',
    **kwargs
  )


def cnn128(**kwargs):
  """
    CNN-128

    Conv: [16, 32, 64, 128, 256, 512]
    
    Local: [2048, 512]
  """
  return cnn(
    conv=[16, 32, 64, 128, 256, 512],
    local=[2048, 512],
    name='cnn128',
    **kwargs
  )


def cnn256(**kwargs):
  """
    CNN-256

    Conv: [16, 32, 64, 128, 256, 512, 1024]
    
    Local: [4096, 1024]
  """
  return cnn(
    conv=[16, 32, 64, 128, 256, 512, 1024],
    local=[4096, 1024],
    name='cnn256',
    **kwargs
  )


# test part
if __name__ == "__main__":
  mod = cnn64(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10}, built=True)
  mod.summary()
