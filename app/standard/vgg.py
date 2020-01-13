# -*- coding: utf-8 -*-
"""VGG

  File: 
    /hat/app/standard/vgg

  Description: 
    VGG模型，包含:
    1. vgg基类
    2. vgg11
    3. vgg13
    4. vgg16
    5. vgg19
    *基于Network_v2
"""


import hat


class base_vgg(hat.Network):
  """VGG基类

    Description:
      VGG模型的基类，包含VGG卷积层和全连接层

    Args:
      times: 每个VGG层的重复次数
      name: Str, optional. 模型名字

    Overrided:
      args: 存放参数
      build: 定义了`keras.Model`并返回
  """
  def __init__(
      self,
      times,
      kernel_size=3,
      pool_size=3,
      pool_step=2,
      local=[4096, 4096],
      drop=0.3,
      resulution=None,
      name='',
      **kwargs):
    self.times = times
    self.kernel_size = kernel_size
    self.pool_size = pool_size
    self.pool_step = pool_step
    self.local = local
    self.drop = drop
    self.resolution = resulution
    self.name = name
    super().__init__(**kwargs)

  def args(self):
    self.conv = [64 * (2 ** i) for i in range(5)]

  def build(self):
    inputs = self.nn.input(self.input_shape)
    x = inputs
    if self.resolution is not None:
      x = self.nn.resolutionscal2d(self.resolution)(x)
    for i in list(zip(self.times, self.conv)):
      x = self.nn.repeat(self.nn.conv, *i, self.kernel_size)(x)
      x = self.nn.maxpool(self.pool_size, self.pool_step)(x)
    x = self.nn.flatten()(x)
    for i in self.local:
      x = self.nn.local(i)(x)
      x = self.nn.dropout(self.drop)(x)
    x = self.nn.dense(self.output_class, activation='softmax')(x)
    return self.nn.model(inputs, x)


def vgg11(**kwargs):
  """VGG-11

    Description:
      times: [1, 1, 2, 2, 2]

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return base_vgg(
      times=[1, 1, 2, 2, 2],
      name='vgg11',
      **kwargs)


def vgg13(**kwargs):
  """VGG-13

    Description:
      times: [2, 2, 2, 2, 2]

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return base_vgg(
      times=[2, 2, 2, 2, 2],
      name='vgg13',
      **kwargs)


def vgg16(**kwargs):
  """VGG-16

    Description:
      times: [2, 2, 3, 3, 3]

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return base_vgg(
      times=[2, 2, 3, 3, 3],
      name='vgg16',
      **kwargs)


def vgg19(**kwargs):
  """VGG-19

    Description:
      times: [2, 2, 4, 4, 4]

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return base_vgg(
      times=[2, 2, 4, 4, 4],
      name='vgg19',
      **kwargs)


# test part
if __name__ == "__main__":
  t = hat.util.Tc()
  t.data.input_shape = (32, 32, 3)
  t.data.output_shape = (10,)
  mod = vgg16(config=t)
  mod.summary()

