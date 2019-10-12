"""
  Google Efficient Net B0~7
  
  ImageNet:
    B0:
    Total params:         5,330,564
    Trainable params:     5,288,548
    Non-trainable params: 42,016
"""

# pylint: disable=no-name-in-module

import math

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import ZeroPadding2D, Cropping2D
from tensorflow.python.keras.optimizers import SGD
from hat.models.advance import AdvNet, ENCI, ENDI


# import setting
__all__ = [
  'enet',
  'enetb0',
  'enetb1',
  'enetb2',
  'enetb3',
  'enetb4',
  'enetb5',
  'enetb6',
  'enetb7',
]


class enet(AdvNet):
  """
    Efficient Net
  """
  def __init__(self, width_coefficient=1, depth_coefficient=1, resolution=224, drop=0.2,
        depth_divisor=8, min_depth=None, id_skip=True, batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3, name='', **kwargs):
    
    self.width_coefficient = width_coefficient
    self.depth_coefficient = depth_coefficient
    self.resolution = resolution
    self.drop = drop
    self.depth_divisor = depth_divisor
    self.min_depth = min_depth
    self.id_skip = id_skip
    self.batch_norm_momentum = batch_norm_momentum
    self.batch_norm_epsilon = batch_norm_epsilon
    self.name = name
    
    self.axis = -1
    super().__init__(**kwargs)

  def args(self):

    self.width_coefficient = self.width_coefficient
    self.depth_coefficient = self.depth_coefficient
    self.resolution = self.resolution
    self.drop = self.drop
    self.depth_divisor = self.depth_divisor
    self.min_depth = self.min_depth
    self.id_skip = self.id_skip
    self.batch_norm_momentum = self.batch_norm_momentum
    self.batch_norm_epsilon = self.batch_norm_epsilon
    self.name = self.name

    self.STEM_CONV = 32
    self.STEM_STEP = 2
    self.HEAD_CONV = 1280
    
    self.DROP_CONNECT = 0
    self.SE_RATE = 4

    self.MB_TIME = [1 , 2 , 2 , 3 , 3  , 4  , 1  ]
    self.MB_CONV = [16, 24, 40, 80, 112, 192, 320]
    self.MB_SIZE = [3 , 3 , 5 , 3 , 5  , 5  , 3  ]
    self.MB_STEP = [1 , 2 , 2 , 2 , 1  , 2  , 1  ]
    self.MB_EXPD = [1 , 6 , 6 , 6 , 6  , 6  , 6  ]
    
    self.OPT = 'adam' # SGD(lr=1e-2, momentum=.9, decay=5e-4)

  def build_model(self):

    x_in = self.input(self.INPUT_SHAPE)

    x = self._fix_input(x_in)

    # Stem part
    x = self.conv(
      x,
      filters=self._round_filters(self.STEM_CONV, self.width_coefficient, self.depth_divisor, self.min_depth),
      kernel_size=3,
      strides=self.STEM_STEP,
      kernel_initializer=ENCI(),
      padding='same',
      use_bias=False)
    x = self.bn(
      x,
      axis=self.axis,
      momentum=self.batch_norm_momentum,
      epsilon=self.batch_norm_epsilon)
    x = self.swish(x)

    # MBConv part
    blocks_list = list(zip(
      self.MB_TIME,
      self.MB_CONV,
      self.MB_SIZE,
      self.MB_STEP,
      self.MB_EXPD))
    drop_connect_rate_per_block = self.DROP_CONNECT / float(sum(self.MB_TIME))
    for block_idx, i in enumerate(blocks_list):
      x = self.MBConv(x, *i, drop_connect_rate_per_block*block_idx)

    # Head part
    x = self.conv(
      x,
      filters=self._round_filters(self.HEAD_CONV, self.width_coefficient, self.depth_divisor, self.min_depth),
      kernel_size=1,
      strides=1,
      kernel_initializer=ENCI(),
      padding='same',
      use_bias=False)
    x = self.bn(
      x,
      axis=self.axis,
      momentum=self.batch_norm_momentum,
      epsilon=self.batch_norm_epsilon)
    x = self.swish(x)

    # output part
    x = self.GAPool(x)
    if self.drop:
      x = self.dropout(x, self.drop)
    x = self.local(
      x,
      self.NUM_CLASSES,
      kernel_initializer=ENDI(),
      activation='softmax'
    )

    return self.Model(inputs=x_in, outputs=x, name='enet')

  def _fix_input(self, x_in):
    """
      fix the input shape to resolution
    """
    h, w = K.int_shape(x_in)[1:3]
    dh = self.resolution - h
    dw = self.resolution - w
    dhh = dh // 2
    dhw = dw // 2

    if dhh >= 0 and dhw > 0:
      x = ZeroPadding2D(padding=((dhh, dh - dhh), (dhw, dw - dhw)))(x_in)
    if dhh < 0 and dhw < 0:
      x = Cropping2D(cropping=((-dhh, dhh - dh), (-dhw, dhw - dw)))(x_in)
    else:
      x = x_in
    return x
  
  def _round_repeats(self, repeats, depth_coefficient):
    """
      Round number of repeats based on depth multiplier.
    """

    if not depth_coefficient:
      return repeats

    return int(math.ceil(depth_coefficient * repeats))

  def _round_filters(self, filters, width_coefficient, depth_divisor, min_depth):
    """
      Round number of filters based on depth multiplier.
    """

    if not width_coefficient:
      return filters

    filters *= width_coefficient
    min_depth = min_depth or depth_divisor
    new_filters = max(
      min_depth,
      int(filters + depth_divisor / 2) // depth_divisor*depth_divisor
    )
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor

    return int(new_filters)

  def MBConvBlock(self, x_in, filters, kernel_size, strides, expand_ratio, drop_connect_rate):
    """
      A class of MBConv: Mobile Inverted Residual Bottleneck.
    """
    input_filters = K.int_shape(x_in)[self.axis]
    expd_filters = input_filters * expand_ratio
    kernel_size = (kernel_size,) * 2 if type(kernel_size) == int else kernel_size
    strides = (strides,) * 2 if type(strides) == int else strides

    x = x_in

    # Expand part
    if expand_ratio != 1:
      x = self.conv(
        x_in,
        expd_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=ENCI(),
        padding='same',
        use_bias=False
      )
      x = self.bn(
        x,
        axis=self.axis,
        momentum=self.batch_norm_momentum,
        epsilon=self.batch_norm_epsilon
      )
      x = self.swish(x)

    # DWConv part
    x = self.dwconv(
      x,
      kernel_size,
      strides=strides,
      depthwise_initializer=ENCI(),
      padding='same',
      use_bias=False
    )
    x = self.bn(
      x,
      axis=self.axis,
      momentum=self.batch_norm_momentum,
      epsilon=self.batch_norm_epsilon
    )
    x = self.swish(x)

    # SE part
    if self.SE_RATE:
      x = self.SE(x, input_filters=input_filters, rate=self.SE_RATE, kernel_initializer=ENDI())

    # Output part
    x = self.conv(
      x,
      filters,
      kernel_size=1,
      strides=1,
      kernel_initializer=ENCI(),
      padding='same',
      use_bias=False
    )
    x = self.bn(
      x,
      axis=self.axis,
      momentum=self.batch_norm_momentum,
      epsilon=self.batch_norm_epsilon
    )

    # skip part
    if self.id_skip:
      if all(s == 1 for s in strides) and input_filters == filters:
        # only apply drop_connect if skip presents.
        if drop_connect_rate:
          x = self.dropconnect(x, drop_connect_rate)
        x = self.add([x, x_in])

    return x

  def MBConv(self, x_in, n, filters, kernel_size, strides, expand_ratio, drop_connect_rate):
    """
      A Stage of MBConv
    """
    n = self._round_repeats(n, self.depth_coefficient)
    filters = self._round_filters(filters, self.width_coefficient, self.depth_divisor, self.min_depth)
    x = self.MBConvBlock(x_in, filters, kernel_size, strides, expand_ratio, drop_connect_rate)
    x = self.repeat(self.MBConvBlock, n-1, filters, kernel_size, 1, expand_ratio, drop_connect_rate)(x)
    return x


def enetb0(**kwargs):
  """
    Effecient Net B0

      width: 1.0
      depth: 1.0
      r    : 224
      drop : 0.2
  """
  return enet(
    width_coefficient=1.0,
    depth_coefficient=1.0,
    resolution=224,
    drop=0.2,
    depth_divisor=8,
    min_depth=None,
    id_skip=True,
    batch_norm_momentum=0.99,
    batch_norm_epsilon=1e-3,
    name='enetb0',
    **kwargs
  )


def enetb1(**kwargs):
  """
    Effecient Net B1

      width: 1.0
      depth: 1.1
      r    : 240
      drop : 0.2
  """
  return enet(
    width_coefficient=1.0,
    depth_coefficient=1.1,
    resolution=240,
    drop=0.2,
    depth_divisor=8,
    min_depth=None,
    id_skip=True,
    batch_norm_momentum=0.99,
    batch_norm_epsilon=1e-3,
    name='enetb1',
    **kwargs
  )


def enetb2(**kwargs):
  """
    Effecient Net B2

      width: 1.1
      depth: 1.2
      r    : 260
      drop : 0.3
  """
  return enet(
    width_coefficient=1.1,
    depth_coefficient=1.2,
    resolution=260,
    drop=0.3,
    depth_divisor=8,
    min_depth=None,
    id_skip=True,
    batch_norm_momentum=0.99,
    batch_norm_epsilon=1e-3,
    name='enetb2',
    **kwargs
  )


def enetb3(**kwargs):
  """
    Effecient Net B3

      width: 1.2
      depth: 1.4
      r    : 300
      drop : 0.3
  """
  return enet(
    width_coefficient=1.2,
    depth_coefficient=1.4,
    resolution=300,
    drop=0.3,
    depth_divisor=8,
    min_depth=None,
    id_skip=True,
    batch_norm_momentum=0.99,
    batch_norm_epsilon=1e-3,
    name='enetb3',
    **kwargs
  )


def enetb4(**kwargs):
  """
    Effecient Net B4

      width: 1.4
      depth: 1.8
      r    : 380
      drop : 0.4
  """
  return enet(
    width_coefficient=1.4,
    depth_coefficient=1.8,
    resolution=380,
    drop=0.4,
    depth_divisor=8,
    min_depth=None,
    id_skip=True,
    batch_norm_momentum=0.99,
    batch_norm_epsilon=1e-3,
    name='enetb4',
    **kwargs
  )


def enetb5(**kwargs):
  """
    Effecient Net B5

      width: 1.6
      depth: 2.2
      r    : 456
      drop : 0.4
  """
  return enet(
    width_coefficient=1.6,
    depth_coefficient=2.2,
    resolution=456,
    drop=0.4,
    depth_divisor=8,
    min_depth=None,
    id_skip=True,
    batch_norm_momentum=0.99,
    batch_norm_epsilon=1e-3,
    name='enetb5',
    **kwargs
  )


def enetb6(**kwargs):
  """
    Effecient Net B6

      width: 1.8
      depth: 2.6
      r    : 528
      drop : 0.5
  """
  return enet(
    width_coefficient=1.8,
    depth_coefficient=2.6,
    resolution=528,
    drop=0.5,
    depth_divisor=8,
    min_depth=None,
    id_skip=True,
    batch_norm_momentum=0.99,
    batch_norm_epsilon=1e-3,
    name='enetb6',
    **kwargs
  )


def enetb7(**kwargs):
  """
    Effecient Net B7

      width: 2.0
      depth: 3.1
      r    : 600
      drop : 0.5
  """
  return enet(
    width_coefficient=2.0,
    depth_coefficient=3.1,
    resolution=600,
    drop=0.5,
    depth_divisor=8,
    min_depth=None,
    id_skip=True,
    batch_norm_momentum=0.99,
    batch_norm_epsilon=1e-3,
    name='enetb7',
    **kwargs
  )


if __name__ == "__main__":
  mod = enetb0(DATAINFO={'INPUT_SHAPE': (224, 224, 3), 'NUM_CLASSES': 1000}, built=True)
  mod.summary()

  from tensorflow.python.keras.utils import plot_model

  plot_model(mod.model,
            to_file='Enet.jpg',
            show_shapes=True)
