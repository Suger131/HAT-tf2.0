"""
  Google Efficient Net B0
"""

# pylint: disable=no-name-in-module
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model

from models.network import NetWork
from models.advanced import AdvNet, Swish, DropConnect, ENCI, ENDI
from utils.counter import Counter


# func
def swish(x, **kwargs):
  x = Swish(name=f"Swish_{Counter('swish')}", **kwargs)(x)
  return x


def dropconnect(x, drop_connect_rate=0, **kwargs):
  x = DropConnect(drop_connect_rate, name=f"DropConnect_{Counter('dropconnect')}", **kwargs)(x)
  return x


class enet(NetWork, AdvNet):
  """
    Efficient Net B0
  """

  def args(self):
    self.STEM_CONV = 32
    self.STEM_STEP = 2
    self.HEAD_CONV = 1280
    self.DROP_CONNECT = 0
    self.D = 0.2
    self.SE_RATE = 4
    self.MB_INDX = [0 , 1 , 2 , 3 , 4  , 5  , 6  ]
    self.MB_CONV = [16, 24, 40, 80, 112, 192, 320]
    self.MB_SIZE = [3 , 3 , 5 , 3 , 5  , 5  , 3  ]
    self.MB_STEP = [1 , 2 , 2 , 1 , 2  , 2  , 1  ]
    self.MB_TIME = [1 , 2 , 2 , 3 , 3  , 4  , 1  ]
    self.MB_EXPD = [1 , 6 , 6 , 6 , 6  , 6  , 6  ]
    

  def build_model(self):

    self.axis = -1
    batch_norm_momentum = 0.99
    batch_norm_epsilon = 1e-3
    
    x_in = self.input(self.INPUT_SHAPE)

    x = x_in

    # Stem part
    x = self.conv(
      x,
      self.STEM_CONV,
      kernel_size=3,
      strides=self.STEM_STEP,
      kernel_initializer=ENCI(),
      padding='same',
      use_bias=False
    )
    x = self.bn(
      x,
      axis=self.axis,
      momentum=batch_norm_momentum,
      epsilon=batch_norm_epsilon
    )
    x = swish(x)

    # blocks part
    blocks_list = list(zip(
      self.MB_INDX,
      self.MB_CONV,
      self.MB_SIZE,
      self.MB_STEP,
      self.MB_TIME,
      self.MB_EXPD))
    drop_connect_rate_per_block = self.DROP_CONNECT / float(sum(self.MB_TIME))
    
    for block_idx, filters, kernel_size, strides, n, expand_ratio in blocks_list:
      x = self.MBConv(
        n, x, filters,
        kernel_size, strides,
        expand_ratio, self.SE_RATE,
        True, drop_connect_rate_per_block * block_idx,
        batch_norm_momentum, batch_norm_epsilon)

    # Head part
    x = self.conv(
      x,
      self.HEAD_CONV,
      kernel_size=1,
      strides=1,
      kernel_initializer=ENCI(),
      padding='same',
      use_bias=False
    )
    x = self.bn(
      x,
      axis=self.axis,
      momentum=batch_norm_momentum,
      epsilon=batch_norm_epsilon
    )
    x = swish(x)

    # output part
    x = self.GAPool(x)
    if self.D:
      x = self.dropout(x, self.D)
    x = self.local(
      x,
      self.NUM_CLASSES,
      kernel_initializer=ENDI(),
      activation='softmax'
    )

    self.model = Model(inputs=x_in, outputs=x, name='enet')

  def MBConv(self,
        n, x_in, filters, kernel_size, strides, expand_ratio, se_rate,
        id_skip, drop_connect_rate, batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3):
    x = x_in
    for i in range(n):
      x = self.MBConvBlock(x, filters, kernel_size, strides, expand_ratio, se_rate,
        id_skip, drop_connect_rate, batch_norm_momentum, batch_norm_epsilon)
      if n > 1:
        strides = 1
    return x

  def MBConvBlock(self, x_in, filters, kernel_size, strides, expand_ratio, se_rate,
                  id_skip, drop_connect_rate, batch_norm_momentum=0.99,
                  batch_norm_epsilon=1e-3):

    input_filters = K.int_shape(x_in)[self.axis]
    _filters = input_filters * expand_ratio
    kernel_size = (kernel_size,) * 2 if type(kernel_size) == int else kernel_size
    strides = (strides,) * 2 if type(strides) == int else strides
    
    # Expand part
    if expand_ratio != 1:
      x = self.conv(
        x_in,
        _filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=ENCI(),
        padding='same',
        use_bias=False
      )
      x = self.bn(
        x,
        axis=self.axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon
      )
      x = swish(x)
    else:
      x = x_in
    
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
      momentum=batch_norm_momentum,
      epsilon=batch_norm_epsilon
    )
    x = swish(x)

    if se_rate:
      x = self.SE(x, input_filters=input_filters, rate=se_rate, kernel_initializer=ENDI())

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
      momentum=batch_norm_momentum,
      epsilon=batch_norm_epsilon
    )

    # skip part
    if id_skip:
      if all(s == 1 for s in strides) and input_filters == filters:
        # only apply drop_connect if skip presents.
        if drop_connect_rate:
          x = dropconnect(x, drop_connect_rate)
        x = self.add([x, x_in])

    return x


# test part
if __name__ == "__main__":
  mod = enet(DATAINFO={'INPUT_SHAPE': (224, 224, 3), 'NUM_CLASSES': 1000})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())

  from tensorflow.python.keras.utils import plot_model

  # plot_model(
  #   mod.model,
  #   to_file=f'enet.png',
  #   show_shapes=True
  # )
