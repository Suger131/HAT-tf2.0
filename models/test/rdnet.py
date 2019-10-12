"""
  RD-Net
  Res&Dense Net
  NOTE:
    默认输入尺寸是(None, 224, 224, 3)
  本模型默认总参数量[参考基准：ImageNet]：
    Total params:           510,154
    Trainable params:       506,986
    Non-trainable params:   3,168
"""


# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import
# pylint: disable=arguments-differ

import math
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.optimizers import SGD
from hat.models.advance import AdvNet


class rdnet(AdvNet):
  """
    RD-Net
  """
  def args(self):
    self.width_coefficient = 0.5
    self.depth_coefficient = 1
    self.resolution = 100
    self.depth_divisor = 8
    self.min_depth = None
    self.drop = 0.5
    self.theta = 0.5
    self.se = int(16 * self.width_coefficient)
    self.group = 0# int(16 * self.width_coefficient)
    self.use_resolution = True
    self.use_bias = False

    self._enable = True

    self.STEM_CONV = 64
    self.STEM_SIZE = 7
    self.STEM_STEP = 2
    self.HEAD_CONV = 2048
    self.HEAD_SIZE = 1
    self.HEAD_STEP = 1

    self.OPT = SGD(lr=0.01, momentum=.9, decay=4e-4)

  def build_model(self):

    x_in = self.input(self.INPUT_SHAPE)

    x = x_in

    # Stem part
    x = self.Stem(x)

    # RBConv Part
    x = self.MBConv(x, 'R', 4, 32, 3, 1)
    x = self.MBConv(x, 'R', 6, 64, 5, 2)
    x = self.MBConv(x, 'R', 5, 128, 3, 2)
    # DBConv Part
    x = self.MBConv(x, 'D', 4, 256, 3, 1)
    x = self.MBConv(x, 'D', 8, 1024, 5, 2)
    x = self.MBConv(x, 'D', 7, 2048, 3, 2)

    # Head part
    x = self.Head(x)

    # Output Part
    x = self.Gmixpool(x)
    if self.drop:
      x = self.dropout(x, self.drop)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='rdnet')

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
    return int(math.ceil(self.depth_coefficient * repeats))

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

  def bn(self, x, *args, **kwargs):
    """Custom BN"""
    if self._enable:
      return super().bn(x, *args, **kwargs)
    else:
      return x

  def relu(self, x, *args, **kwargs):
    """Custom ReLU"""
    if self._enable:
      return super().relu(x, *args, **kwargs)
    else:
      return x

  def Stem(self, x_in):
    i1 = self.INPUT_SHAPE[0]
    filters = self.round_filters(self.STEM_CONV, 32)
    x = x_in

    if self.use_resolution and i1 < self.resolution:
      s1 = math.ceil(self.resolution / i1)
      s2 = math.floor(self.resolution / i1)
      s = s1 if abs(self.resolution - i1 * s1 - 7 + s1) < abs(self.resolution - i1 * s2 - 7 + s2) else s2
      x = Conv2DTranspose(self.INPUT_SHAPE[-1], 7, strides=s, padding='valid')(x)
      x = self.bn(x)
      x = self.proc_input(x, self.resolution)
    
    x = self.conv(
      x,
      filters, # self.STEM_CONV,
      self.STEM_SIZE,
      self.STEM_STEP,
      use_bias=self.use_bias)
    x = self.bn(x)
    x = self.relu(x)
    return x

  def Head(self, x_in):
    min_filters = int(math.pow(2, math.ceil(math.log2(self.NUM_CLASSES))))
    filters = self.round_filters(self.HEAD_CONV, min_filters)
    x = self.conv(
      x_in,
      filters,
      self.HEAD_SIZE,
      self.HEAD_STEP,
      use_bias=self.use_bias)
    x = self.bn(x)
    x = self.relu(x)
    return x

  def NBlock(self, x_in, filters, kernel_size, strides=1, group=None):
    """MBlock"""
    group = group or self.group
    x = x_in
    # SE Part
    if self.se:
      x = self.SE(x, self.se)
    # Conv Part
    x = self.conv(x, filters, 1, 1, use_bias=self.use_bias)
    x = self.bn(x)
    x = self.relu(x)
    # Group Conv Part
    if group and not filters % group:
      x = self.Gconv(x, group, kernel_size=3, activation=None, use_group_bias=True)
      x = self.bn(x)
      x = self.relu(x)
    # DwConv Part
    x = self.dwconv(x, kernel_size, strides, use_bias=self.use_bias)
    x = self.bn(x)
    return x

  def RBlock(self, x_in, filters, kernel_size):
    """RBlock"""
    x = x_in
    x = self.NBlock(x, filters, kernel_size)
    x = self.add([x_in, x])
    return x

  def DBlock(self, x_in, filters, kernel_size):
    """DBlock"""
    x = x_in
    x = self.NBlock(x, filters, kernel_size)
    x = self.concat([x_in, x])
    return x

  def Transition(self, x_in, filters, kernel_size, strides=1):
    """Transition"""
    channels = K.int_shape(x_in)[-1]
    x = x_in
    x = self.NBlock(x, filters, kernel_size, strides)
    x_aux = x_in
    if filters != channels:
      x_aux = self.conv(x_aux, filters, 1, 1)
    if strides != 1:
      x_aux = self.mixpool(x_aux, strides)
    if filters != channels or strides != 1:
      x_aux = self.bn(x_aux)
      x = self.add([x, x_aux])
    return x

  def RBConv(self, x_in, n, filters, kernel_size, strides=1):
    """RBConv"""
    n = self.round_repeats(n)
    filters = self.round_filters(filters)
    x = x_in
    x = self.Transition(x, filters, kernel_size, strides)
    x = self.repeat(self.RBlock, n - 1, filters, kernel_size)(x)
    return x

  def DBConv(self, x_in, n, filters, kernel_size, strides=1, theta=None):
    """DBConv"""
    n = self.round_repeats(n)
    filters = self.round_filters(filters)
    theta = theta or self.theta
    nfilters = int(K.int_shape(x_in)[-1] * theta)
    dfilters = filters - nfilters
    afilters = dfilters // (n - 1)
    bfilters = dfilters - (n - 2) * afilters or afilters
    x = x_in
    x = self.Transition(x, nfilters, kernel_size, strides)
    x = self.repeat(self.DBlock, n - 2, afilters, kernel_size)(x)
    if n - 1:
      x = self.DBlock(x, bfilters, kernel_size)
    return x

  def MBConv(self, x_in, mode, n, filters, kernel_size, strides=1, theta=None):
    """Muti-Block-Conv"""
    x = x_in
    if mode == 'R':
      x = self.RBConv(x, n, filters, kernel_size, strides)
    elif mode == 'D':
      x = self.DBConv(x, n, filters, kernel_size, strides, theta)
    return x


# test
if __name__ == "__main__":
  mod = rdnet(DATAINFO={
    'INPUT_SHAPE': (32, 32, 3),
    'NUM_CLASSES': 10}, built=True)
  mod.summary()
  # mod.flops()

  from tensorflow.python.keras.utils import plot_model

  plot_model(mod.model,
            to_file='rdnet.jpg',
            show_shapes=True)

