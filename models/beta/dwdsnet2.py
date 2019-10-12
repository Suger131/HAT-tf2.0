"""
  DwDsNet2
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


class dwdsnet2(AdvNet):
  """
    DwDsNet
  """

  def args(self):
    self.CONV_BIAS = False
    self.SE_R = 16
    self.DROP = 0.2
    self.THETA = 0.5

    self.STEM_CONV = 48
    self.STEM_SIZE = 7
    self.STEM_STEP = 2
    self.HEAD_CONV = 1024
    self.HEAD_SIZE = 1
    self.HEAD_STEP = 1

    self.DB_TIME = [ 2,  3,  4,  4,  4,  5]
    self.DB_CONV = [16, 16, 24, 48, 96, 80]
    self.DB_SIZE = [ 3,  3,  3,  3,  3,  3]
    self.DB_STEP = [ 1,  2,  1,  2,  2,  2]
    self.DB_THET = [1/3, 0,  0,  0,  0,  0]
    
    # self.OPT = SGD(lr=0.1, momentum=.9, decay=5e-4)
    # self.OPT = Adam(lr=5e-2, decay=2e-4)
    self.OPT = Adam()

  def build_model(self):

    self.axis = -1
    x_in = self.input(self.INPUT_SHAPE)
    
    # Stem Part
    x = self.Stem(x_in)

    # DB Part
    blocks_list = list(zip(
      self.DB_TIME,
      self.DB_CONV,
      self.DB_SIZE,
      self.DB_STEP,
      self.DB_THET,))
    for i in blocks_list:
      x = self.DBlock(x, *i)
    
    # x = self.DBlock(x, 2, 16, 3, 1, theta=1 / 3)
    # x = self.DBlock(x, 3, 16, 3, 2)
    # x = self.DBlock(x, 4, 24, 3, 1)
    # x = self.DBlock(x, 4, 48, 3, 2)
    # x = self.DBlock(x, 4, 96, 3, 2)
    # x = self.DBlock(x, 5, 80, 3, 2)
    
    # Head Part
    x = self.Head(x)

    # Output Part
    x = self.Gmixpool(x)
    if self.DROP:
      x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='dwdsnet')

  def round_filters(self, filters):
    return filters

  def round_repeats(self, repeats):
    return repeats

  def mixpool(self, x_in, pool_size=2, strides=2):
    x1 = self.maxpool(x_in, pool_size, strides)
    x2 = self.avgpool(x_in, pool_size, strides)
    x = self.add([x1, x2])
    return x

  def Gmixpool(self, x_in):
    x1 = self.GAPool(x_in)
    x2 = self.GMPool(x_in)
    x = self.add([x1, x2])
    return x

  def Stem(self, x_in):
    filters = self.round_filters(self.STEM_CONV)
    x = self.conv(
      x_in,
      filters,
      self.STEM_SIZE,
      self.STEM_STEP,
      use_bias=self.CONV_BIAS)
    x = self.bn(x)
    x = self.relu(x)
    return x

  def Head(self, x_in):
    filters = self.round_filters(self.HEAD_CONV)
    x = self.conv(
      x_in,
      filters,
      self.HEAD_SIZE,
      self.HEAD_STEP,
      use_bias=self.CONV_BIAS)
    x = self.bn(x)
    x = self.relu(x)
    return x

  def Dense(self, x_in, filters, kernel_size, strides=1, skip=False, theta=1, group=0):
    """
      Dense Block
    """
    nfilters = int(K.int_shape(x_in)[self.axis] * theta)
    x = x_in
    # Group Conv Part
    if group:
      x = self.Gconv(x, group, kernel_size=3, activation=None, use_group_bias=True)
      x = self.bn(x)
      x = self.relu(x)
    # DWConv Part
    x = self.dwconv(x, kernel_size, strides, use_bias=self.CONV_BIAS)
    x = self.bn(x)
    x = self.relu(x)
    # SE Part
    if self.SE_R:
      x = self.SE(x, self.SE_R)
    # Output Part
    if skip:
      x = self.conv(x, nfilters, 1, 1, use_bias=self.CONV_BIAS)
      x = self.bn(x)
      x_aux = self.conv(x_in, nfilters, 1, 1, use_bias=self.CONV_BIAS)
      if strides != 1:
        x_aux = self.mixpool(x_aux, 3, 2)
      x = self.add([x, x_aux])
    else:
      x = self.conv(x, filters, 1, 1, use_bias=self.CONV_BIAS)
      x = self.bn(x)
      x = self.concat([x_in, x])
    return x

  def DBlock(self, x_in, n, filters, kernel_size, strides=1, theta=0, group=0):
    """DBlock"""
    n = self.round_repeats(n)
    filters = self.round_filters(filters)
    theta = theta or self.THETA
    x = self.Dense(x_in, filters, kernel_size, strides, skip=True, theta=theta, group=group)
    x = self.repeat(self.Dense, n-1, filters, kernel_size, 1, group=group)(x)
    return x


# test part
if __name__ == "__main__":
  mod = dwdsnet2(DATAINFO={'INPUT_SHAPE': (224, 224, 3), 'NUM_CLASSES': 1000}, built=True)
  mod.summary()
  
  from tensorflow.python.keras.utils import plot_model

  plot_model(mod.model,
            to_file='dwdsnet.jpg',
            show_shapes=True)
