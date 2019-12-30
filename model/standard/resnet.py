# -*- coding: utf-8 -*-
"""ResNet

  File: 
    /hat/model/standard/resnet

  Description: 
    ResNet模型，包含:
    1. resnet基类
    *基于Network_v2
"""


import hat


# import setting
__all__ = [
  'resnet',
  'resnet50',
  'resnet101',
  'resnet152',
  'resnetse50',
  'resnetse101',
  'resnetse152',
  'resnext50',
  'resnext101',
  'resnext152',
]


class resnet(hat.Network):
  """ResNet基类

    Description:
      1

    Attributes:
      1

    Overrided:
      args: 存放参数
      build: 定义了`keras.Model`并返回
  """
  def __init__(
      self,
      times,
      use_se=False,
      use_group=False,
      kernel_size=3,
      pool_size=3,
      pool_step=2,
      drop=0.2,
      se_ratio=1/16,
      resulution=None,
      name='',
      **kwargs):
    self.times = times
    self.use_se = use_se
    self.use_group = use_group
    self.kernel_size = kernel_size
    self.pool_size = pool_size
    self.pool_step = pool_step
    self.drop = drop
    self.se_ratio = se_ratio
    self.resolution = resulution
    self.name = name
    super().__init__(**kwargs)

  def args(self):
    self.first_conv = 64
    self.first_kernel_size = 7
    self.first_strides = 2
    self.res_filters = [256 * (2 ** i) for i in range(4)]
    self.res_strides = [1, 2, 2, 2]
    if self.config.input_shape[0] // 32 < 4:
      self.res_strides[3] = 1
    self.group = 32

  def build(self):
    inputs = self.nn.input(self.input_shape)
    x = inputs
    if self.resolution is not None:
      x = self.nn.resolutionscal2d(self.resolution)(x)
    x = self.nn.conv(self.first_conv, self.first_kernel_size, 
        strides=self.first_strides, activation='relu')(x)
    x = self.nn.maxpool(self.pool_size, self.pool_step)(x)
    for i in list(zip(self.times, self.res_filters, self.res_strides)):
      x = self.block(*i)(x)
    x = self.nn.bn()(x)
    x = self.nn.relu()(x)
    x = self.nn.gapool()(x)
    if self.drop:
      x = self.nn.dropout(self.drop)(x)
    x = self.nn.dense(self.output_class, activation='softmax')(x)
    return self.nn.model(inputs, x)

  def block(self, times, filters, strides=1):
    def block_layer(inputs):
      x = self.bottle(filters, strides=strides, start=True)(inputs)
      return self.nn.repeat(self.bottle, times - 1, filters)(x)
    return block_layer

  def bottle(self, filters, strides=1, start=False):
    def bottle_layer(inputs):
      x = inputs
      x = self.nn.bn()(x)
      x = self.nn.relu()(x)
      if start:
        x_ = self.nn.conv(filters, 1, activation='relu')(x)
        if strides != 1:
          x_ = self.nn.maxpool(self.pool_size, strides)(x_)
      else:
        x_ = inputs
      if self.use_group:
        x = self.nn.conv(filters // 2, 1, activation='relu')(x)
        x = self.nn.bn()(x)
        x = self.nn.relu()(x)
        x = self.nn.gconv(self.group, kernel_size=self.kernel_size,
            strides=strides, padding='same', activation='relu')(x)
      else:
        x = self.nn.conv(filters // 4, 1, activation='relu')(x)
        x = self.nn.bn()(x)
        x = self.nn.relu()(x)
        x = self.nn.conv(filters // 4, self.kernel_size, strides=strides, 
            activation='relu')(x)
      x = self.nn.bn()(x)
      x = self.nn.relu()(x)
      x = self.nn.conv(filters, 1)(x)
      if self.use_se:
        x = self.nn.se(self.se_ratio)(x)
      return self.nn.add()([x, x_])
    return bottle_layer


def resnet50(**kwargs):
  """ResNet-50

    Description: 
      times: [3, 4, 6, 3]

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return resnet(
    times=[3, 4, 6, 3],
    name='resnet50',
    **kwargs
  )


def resnet101(**kwargs):
  """ResNet-101

    Description: 
      times: [3, 4, 23, 3]

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return resnet(
    times=[3, 4, 23, 3],
    name='resnet101',
    **kwargs
  )


def resnet152(**kwargs):
  """ResNet-152

    Description: 
      times: [3, 8, 36, 3]

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return resnet(
    times=[3, 8, 36, 3],
    name='resnet152',
    **kwargs
  )


def resnetse50(**kwargs):
  """ResNet-SE-50

    Description: 
      times: [3, 4, 6, 3]
      use_se: True

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return resnet(
    times=[3, 4, 6, 3],
    use_se=True,
    name='resnetse50',
    **kwargs
  )


def resnetse101(**kwargs):
  """ResNet-SE-101

    Description: 
      times: [3, 4, 23, 3]
      use_se: True

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return resnet(
    times=[3, 4, 23, 3],
    use_se=True,
    name='resnetse101',
    **kwargs
  )


def resnetse152(**kwargs):
  """ResNet-SE-152

    Description: 
      times: [3, 8, 36, 3]
      use_se: True

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return resnet(
    times=[3, 8, 36, 3],
    use_se=True,
    name='resnetse152',
    **kwargs
  )


def resnext50(**kwargs):
  """ResNeXt-50

    Description: 
      times: [3, 4, 6, 3]
      use_group: True

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return resnet(
    times=[3, 4, 6, 3],
    use_group=True,
    name='resnext50',
    **kwargs
  )


def resnext101(**kwargs):
  """ResNeXt-101

    Description: 
      times: [3, 4, 23, 3]
      use_group: True

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return resnet(
    times=[3, 4, 23, 3],
    use_group=True,
    name='resnext101',
    **kwargs
  )


def resnext152(**kwargs):
  """ResNeXt-152

    Description: 
      times: [3, 8, 36, 3]
      use_group: True

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return resnet(
    times=[3, 8, 36, 3],
    use_group=True,
    name='resnext152',
    **kwargs
  )


# test part
if __name__ == "__main__":
  t = hat.util.Tc()
  t.data.input_shape = (224, 224, 3)
  t.data.output_shape = (1000,)
  mod = resnext50(config=t)
  mod.summary()

