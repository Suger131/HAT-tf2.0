"""
  VGG
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           37,721,154
    Trainable params:       37,712,706
    Non-trainable params:   8,448
"""

# pylint: disable=no-name-in-module

import math
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.layers import Conv2DTranspose
from hat.models.advance import AdvNet


# import setting
__all__ = [
  'vgg',
  'vgg11',
  'vgg13',
  'vgg16',
  'vgg19',
]


class vgg(AdvNet):
  """
    VGG-Net
  """
  
  def __init__(self, times, name='', **kwargs):
    self.TIME = times
    self.NAME = name
    super().__init__(**kwargs)

  def args(self):

    self.TIME = self.TIME
    self.USE_MIN_SIZE = True
    self.MIN_SIZE = 100

    self.CONV = [64, 128, 256, 512, 512]
    self.LOCAL = [4096, 4096]
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

    # conv part
    for i in list(zip(self.TIME, self.CONV)):
      x = self.repeat(self.conv_bn, *i, 3)(x)
      x = self.maxpool(x, 3, 2)

    # local part
    x = self.flatten(x)
    for i in self.LOCAL:
      x = self.local(x, i)
      x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name=self.NAME)


def vgg11(**kwargs):
  """
    VGG-11
    
    Times: [1, 1, 2, 2, 2]
  """
  return vgg(
    times=[1, 1, 2, 2, 2],
    name='vgg11',
    **kwargs
  )


def vgg13(**kwargs):
  """
    VGG-13
    
    Times: [2, 2, 2, 2, 2]
  """
  return vgg(
    times=[2, 2, 2, 2, 2],
    name='vgg13',
    **kwargs
  )


def vgg16(**kwargs):
  """
    VGG-16
    
    Times: [2, 2, 3, 3, 3]
  """
  return vgg(
    times=[2, 2, 3, 3, 3],
    name='vgg16',
    **kwargs
  )


def vgg19(**kwargs):
  """
    VGG-19
    
    Times: [2, 2, 4, 4, 4]
  """
  return vgg(
    times=[2, 2, 4, 4, 4],
    name='vgg19',
    **kwargs
  )


# test part
if __name__ == "__main__":
  mod = vgg16(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10}, built=True)
  mod.summary()
