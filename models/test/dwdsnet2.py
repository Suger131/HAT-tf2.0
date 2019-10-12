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

import math
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.optimizers import *
from hat.models.advance import AdvNet


class dwdsnet2(AdvNet):
  """
    DwDsNet2
  """

  def args(self):

    self.width_coefficient = 1
    self.depth_coefficient = 1
    self.resolution = 100
    self.depth_divisor = 8
    self.min_depth = None
    self.min_repeats = 2
    self.THETA = 0.5
    self.SE_R = 0# int(16 * self.width_coefficient)
    self.GROUP = 0# int(16 * self.width_coefficient)
    self.MIN_SIZE = 100
    self.USE_MIN_SIZE = True
    self.CONV_BIAS = False
    self.DROP = 0.5
    
    self.STEM_CONV = 32
    self.STEM_SIZE = 7
    self.STEM_STEP = 2
    self.HEAD_CONV = 1024
    self.HEAD_SIZE = 1
    self.HEAD_STEP = 1

    self.DB_TIME = [ 2,  4,  4,  4,   4,   4]
    self.DB_CONV = [32, 32, 64, 128, 256, 512]
    self.DB_SIZE = [ 3,  3,  5,  3,   5,   3]
    self.DB_STEP = [ 1,  2,  2,  1,   2,   2]
    self.DB_THET = [ 0,  0,  0,  0,   0,   0]
    
    # self.OPT = Adam()
    self.OPT = SGD(lr=0.01, momentum=.9, decay=4e-4)

  def build_model(self):

    self.axis = -1
    x_in = self.input(self.INPUT_SHAPE)
    x = x_in

    # Stem
    x = self.Stem(x)

    x = self.DBlock(x, 2, 3, 64, 1, tmode='concat')
    x = self.DBlock(x, 4, 3, 256, 2, tmode='concat')
    x = self.DBlock(x, 4, 3, 512, 2, tmode='concat')
    x = self.DBlock(x, 4, 3, 256, 1)
    x = self.DBlock(x, 4, 3, 1024, 2, tmode='concat')
    x = self.DBlock(x, 4, 3, 2048, 2)

    # Head
    x = self.Head(x)

    # Output Part
    x = self.Gmixpool(x)
    if self.DROP:
      x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='dwdsnet2')

  def round_filters(self, filters, min_filters=0):
    """
      Round number of filters.
    """
    filters *= self.width_coefficient
    min_depth = min_filters or self.min_depth or self.depth_divisor

    # 四舍五入，并且限制最小值
    new_filters = max(
      min_depth,
      int(filters + self.depth_divisor / 2) // self.depth_divisor*self.depth_divisor
    )
    # Make sure that round down does not go down by more than 90%.
    if new_filters < 0.9 * filters:
        new_filters += self.depth_divisor
    
    return int(new_filters)

  def round_repeats(self, repeats):
    return int(max(self.min_repeats, math.ceil(self.depth_coefficient * repeats)))

  def mixpool(self, x_in, pool_size=2, strides=None):
    # x1 = self.maxpool(x_in, pool_size, strides)
    # x2 = self.avgpool(x_in, pool_size, strides)
    # x = self.add([x1, x2])
    x = self.maxpool(x_in, pool_size, strides)
    return x

  def Gmixpool(self, x_in):
    # x1 = self.GAPool(x_in)
    # x2 = self.GMPool(x_in)
    # x = self.add([x1, x2])
    x = self.GAPool(x_in)
    return x

  def Stem(self, x_in):
    i1 = self.INPUT_SHAPE[0]
    x = x_in

    if self.USE_MIN_SIZE and i1 < self.MIN_SIZE:
      s1 = math.ceil(self.MIN_SIZE / i1)
      s2 = math.floor(self.MIN_SIZE / i1)
      s = s1 if abs(self.MIN_SIZE - i1 * s1 - 7 + s1) < abs(self.MIN_SIZE - i1 * s2 - 7 + s2) else s2
      x = Conv2DTranspose(self.INPUT_SHAPE[-1], 7, strides=s, padding='valid')(x)#, kernel_initializer=self.KERNEL_INIT
      x = self.proc_input(x, self.MIN_SIZE)

    x = self.conv(
      x,
      self.STEM_CONV,
      self.STEM_SIZE,
      self.STEM_STEP,
      use_bias=self.CONV_BIAS)
    # x = self.bn(x)
    # x = self.relu(x)
    return x

  def Head(self, x_in):
    min_filters = int(math.pow(2, math.ceil(math.log2(self.NUM_CLASSES))))
    filters = self.round_filters(self.HEAD_CONV, min_filters)
    x = x_in
    x = self.conv(
      x_in,
      filters,
      self.HEAD_SIZE,
      self.HEAD_STEP,
      use_bias=self.CONV_BIAS)
    # x = self.bn(x)
    # x = self.relu(x)
    return x

  def Transition(self, x_in, kernel_size, strides=1, theta=1, group=None, mode='add'):
    """Transition"""
    nfilters = int(K.int_shape(x_in)[self.axis] * theta)
    x = x_in
    # Conv Part
    x = self.conv(x, nfilters, 1, 1, use_bias=self.CONV_BIAS)
    # Aux Part
    x_aux = x
    # Group Conv Part
    if group:
      x = self.Gconv(x, group, kernel_size=3, activation=None, use_group_bias=True)
    # DWConv Part
    x = self.dwconv(x, kernel_size, strides, use_bias=self.CONV_BIAS)
    # Output Part
    if strides != 1:
      x_aux = self.mixpool(x_aux, strides)
    if mode == 'add':
      x = self.add([x, x_aux])
    elif mode == 'concat':
      x = self.concat([x, x_aux])
    return x

  def Dense(self, x_in, kernel_size, filters, strides=1, group=None, mode='concat'):
    """Dense"""
    x = x_in
    # SE Part
    if self.SE_R:
      x = self.SE(x, self.SE_R)
    # Conv Part
    x = self.conv(x, filters, 1, 1, use_bias=self.CONV_BIAS)
    # x = self.bn(x)
    # x = self.relu(x)
    # Group Conv Part
    if group:
      x = self.Gconv(x, group, kernel_size=3, activation=None, use_group_bias=True)
      # x = self.bn(x)
      # x = self.relu(x)
    # DWConv Part
    x = self.dwconv(x, kernel_size, strides, use_bias=self.CONV_BIAS)
    # x = self.bn(x)
    # Output Part
    if mode == 'add':
      x = self.add([x_in, x])
    elif mode == 'concat':
      x = self.concat([x_in, x])
    return x

  def DBlock(self, x_in, n, kernel_size, filters, strides=1, theta=None, group=None, tmode='add'):
    """DBlock"""
    n = self.round_repeats(n)
    filters = self.round_filters(filters)
    theta = theta or self.THETA
    group = group or self.GROUP

    x = x_in
    x = self.Transition(x, kernel_size, strides=strides, theta=theta, group=group, mode=tmode)

    nfilters = int(K.int_shape(x)[self.axis])
    if nfilters == filters:
      x = self.repeat(self.Dense, n - 1, kernel_size, filters, group=group, mode='add')(x)
    else:
      mfilters = (filters - nfilters) // (n - 1)
      lfilters = filters - nfilters - (n - 2) * mfilters or mfilters
      x = self.repeat(self.Dense, n - 2, kernel_size, mfilters, group=group)(x)
      x = self.Dense(x, kernel_size, lfilters, group=group)
    return x


# test part
if __name__ == "__main__":
  mod = dwdsnet2(DATAINFO={'INPUT_SHAPE': (224, 224, 3), 'NUM_CLASSES': 1000}, built=True)
  mod.summary()
  # mod.flops()

  from tensorflow.python.keras.utils import plot_model

  plot_model(mod.model,
            to_file='dwdsnet2.jpg',
            show_shapes=True)
