# -*- coding: utf-8 -*-
"""CNN

  File: 
    /hat/model/standard/cnn

  Description: 
    CNN模型，包含:
    1. CNN基类
    2. CNN32
    3. CNN64
    4. CNN128
    5. CNN256
    * 基于Network_v2
"""


import hat


# import setting
__all__ = [
  'cnn',
  'cnn32',
  'cnn64',
  'cnn128',
  'cnn256',
]


class cnn(hat.Network):
  """CNN基类
  
    Description: 
      CNN模型的基类，包含简单卷积层和全连接层

    Attributes:
      conv: List of Int. 一个元素对应一个卷积层，元素的值为卷积层的Channels
      local: List of Int. 一个元素对应一个全连接层，元素的值为全连接层的节点数
      drop: float in [0, 1). 随机失活率
      name: Str, optional. 模型名字

    Overrided:
      args: 存放参数
      build: 定义了`keras.Model`并返回
  """
  def __init__(self, conv, local, drop=0.5, name='', **kwargs):
    self.conv = conv
    self.local = local
    self.drop = drop
    self.name = name
    super().__init__(**kwargs)

  def args(self):
    self.conv_kernel_size = 5
    # import tensorflow as tf
    # self.opt = tf.keras.optimizers.SGD(lr=0.1, momentum=.9)

  def build(self):
    inputs = self.nn.input(self.config.input_shape)
    x = inputs
    for i in self.conv:
      x = self.nn.conv(i, self.conv_kernel_size, activation='relu')(x)
      x = self.nn.maxpool()(x)
    x = self.nn.flatten()(x)
    for i in self.local:
      x = self.nn.dense(i, activation='relu')(x)
      x = self.nn.dropout(self.drop)(x)
    x = self.nn.dense(self.config.output_shape[-1], activation='softmax')(x)
    return self.nn.model(inputs, x)


def cnn32(**kwargs):
  """CNN-32

    Description: 
      Conv: [16, 32, 64, 128]
      Local: [512, 256]

    Args:
      Consistent with Network.

    Return:
      Network.

    Raises:
      None
  """
  return cnn(
    conv=[16, 32, 64, 128],
    local=[512, 256],
    name='cnn32',
    **kwargs
  )


def cnn64(**kwargs):
  """CNN-64

    Description: 
      Conv: [16, 32, 64, 128, 256]
      Local: [1024, 256]

    Args:
      Consistent with Network.

    Return:
      Network.

    Raises:
      None
  """
  return cnn(
    conv=[16, 32, 64, 128, 256],
    local=[1024, 256],
    name='cnn64',
    **kwargs
  )


def cnn128(**kwargs):
  """CNN-128

    Description: 
      Conv: [16, 32, 64, 128, 256, 512]
      Local: [2048, 512]

    Args:
      Consistent with Network.

    Return:
      Network.

    Raises:
      None
  """
  return cnn(
    conv=[16, 32, 64, 128, 256, 512],
    local=[2048, 512],
    name='cnn128',
    **kwargs
  )


def cnn256(**kwargs):
  """CNN-256

    Description: 
      Conv: [16, 32, 64, 128, 256, 512, 1024]
      Local: [4096, 1024]

    Args:
      Consistent with Network.

    Return:
      Network.

    Raises:
      None
  """
  return cnn(
    conv=[16, 32, 64, 128, 256, 512, 1024],
    local=[4096, 1024],
    name='cnn256',
    **kwargs
  )


# test part
if __name__ == '__main__':
  t = hat._TC()
  t.input_shape = (28, 28, 1)
  t.output_shape = (10,)
  mod = cnn32(config=t)
  t.model.summary()
