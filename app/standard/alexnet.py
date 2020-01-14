# -*- coding: utf-8 -*-
"""AlexNet

  File: 
    /hat/app/standard/alexnet

  Description: 
    Alexnet模型，包含:
    1. Alexnet基类
    2. Alexnet
    3. ZFNet
    *基于Network_v2
"""


import hat


class base_alexnet(hat.Network):
  """AlexNet基类
  
    Description: 
      AlexNet模型的基类，包含简单卷积层和全连接层

    Args:
      kernel_size: List of 5 Ints. 一个元素对应一个卷积层，元素的值
          为卷积层的kernel_size
      first_conv_strides: Int. 第一层卷积的步长
      name: Str, optional. 模型名字

    Overrided:
      args: 存放参数
      build: 定义了`keras.Model`并返回
  """
  def __init__(
      self,
      kernel_size=[11, 5, 3, 3, 3],
      first_conv_strides=4,
      conv=[96, 256, 384, 384, 256],
      local=[4096, 4096],
      pool_size=3,
      pool_step=2,
      padding='valid',
      drop=0.5,
      name='',
      **kwargs):
    self.kernel_size = kernel_size
    self.first_conv_strides = first_conv_strides
    self.conv = conv
    self.local = local
    self.pool_size = pool_size
    self.pool_step = pool_step
    self.padding = padding
    self.drop = drop
    self.name = name
    super().__init__(**kwargs)

  def args(self):
    # self.conv = [96, 256, 384, 384, 256]
    # self.local = [4096, 4096]
    # self.kernel_size = [11, 5, 3, 3, 3]
    # self.first_conv_strides = 4
    # self.pool_size = 3
    # self.pool_step = 2
    # self.padding = 'valid'
    # self.drop = 0.5
    pass

  def build(self):
    inputs = self.nn.input(self.input_shape)
    x = inputs

    x = self.nn.conv(self.conv[0], self.kernel_size[0],
        strides=self.first_conv_strides, padding=self.padding,
        activation='relu')(x)
    x = self.nn.bn()(x)
    x = self.nn.maxpool(self.pool_size, self.pool_step)(x)

    x = self.nn.conv(self.conv[1], self.kernel_size[1],
        padding=self.padding, activation='relu')(x)
    x = self.nn.bn()(x)
    x = self.nn.maxpool(self.pool_size, self.pool_step)(x)

    x = self.nn.conv(self.conv[2], self.kernel_size[2],
        padding=self.padding, activation='relu')(x)
    x = self.nn.conv(self.conv[3], self.kernel_size[3],
        padding=self.padding, activation='relu')(x)
    x = self.nn.conv(self.conv[4], self.kernel_size[4],
        padding=self.padding, activation='relu')(x)
    x = self.nn.maxpool(self.pool_size, self.pool_step)(x)

    x = self.nn.flatten()(x)
    x = self.nn.local(self.local[0])(x)
    if self.drop:
      x = self.nn.dropout(self.drop)(x)
    x = self.nn.local(self.local[1])(x)
    if self.drop:
      x = self.nn.dropout(self.drop)(x)
    x = self.nn.dense(self.output_class, activation='softmax')(x)
    
    return self.nn.model(inputs, x)


def alexnet(**kwargs):
  """AlexNet
  
    Description: 
      kernel size: [11, 5, 3, 3, 3]
      first conv strides: 4

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  return base_alexnet(
      kernel_size=[11, 5, 3, 3, 3],
      first_conv_strides=4,
      name='alexnet',
      **kwargs)


def zfnet(**kwargs):
  """ZFNet
  
    Description: 
      kernel size: [7, 5, 3, 3, 3]
      first conv strides: 2

    Args:
      Consistent with hat.Network

    Return:
      hat.Network

    Raises:
      None
  """
  # FIXME: check the `ZFNet` paper
  return base_alexnet(
      kernel_size=[7, 5, 3, 3, 3],
      first_conv_strides=2,
      name='alexnet',
      **kwargs)


# test part
if __name__ == '__main__':
  hat.config.test((224, 224, 3), (1000,))
  mod = alexnet()
  mod.summary()
  mod = zfnet()
  mod.summary()

