# -*- coding: utf-8 -*-
"""MobileNet

  File: 
    /hat/app/standard/mobilenet

  Description: 
    MobileNet模型
    1. mobilenetv2
    *基于Network_v2
"""


import hat


class mobilenetv2(hat.Network):
  """MobileNet V2

    Description:
      MobileNet V2模型

    Args:
      None

    Overrided:
      args: 存放参数
      build: 定义了`keras.Model`并返回
  """
  def args(self):
    self.conv = [32, 640]
    self.irb_time = [1, 2, 3, 4, 3, 2, 1]
    self.irb_conv = [16, 24, 32, 48, 64, 96, 160]
    self.irb_expd = [1, 3, 3, 3, 3, 3, 3]
    self.irb_step = [1, 2, 2, 1, 2, 2, 1]
    self.irb = list(zip(
        self.irb_time,
        self.irb_conv,
        self.irb_expd,
        self.irb_step,))

  def build(self):
    inputs = self.nn.input(self.input_shape)
    x = inputs
    x = self.nn.conv(self.conv[0], 3, 2)(x)
    x = self.nn.bn()(x)
    x = self.nn.relu()(x)

    for i in self.irb:
      x = self.inverted_residual_block(*i)(x)

    x = self.nn.conv(self.conv[1], 1, 1)(x)
    x = self.nn.bn()(x)
    x = self.nn.relu()(x)
    
    x = self.nn.gapool()(x)
    x = self.nn.dense(self.output_class, activation='softmax')(x)
    return self.nn.model(inputs, x)

  def inverted_residual_block(self, n, filters, k, strides):
    def innet_layer(inputs):
      x = self.bottle_neck(filters, k, strides)(inputs)
      x = self.nn.repeat(self.bottle_neck, n - 1, filters, k)(x)
      return x
    return innet_layer

  def bottle_neck(self, filters, k, strides=None):
    def inner_layer(inputs):
      new_channels = self.nn.get_channels(inputs) * k
      x = self.nn.conv(new_channels, 1)(inputs)
      x = self.nn.bn()(x)
      x = self.nn.relu()(x)
      x = self.nn.dwconv(3, strides or 1)(x)
      x = self.nn.bn()(x)
      x = self.nn.relu()(x)
      x = self.nn.conv(filters, 1)(x)
      x = self.nn.bn()(x)
      if strides is not None:
        inputs = self.nn.conv(filters, 3, strides)(inputs)
        inputs = self.nn.bn()(inputs)
      x = self.nn.add()([inputs, x])
      x = self.nn.relu()(x)
      return x
    return inner_layer


# test
if __name__ == "__main__":
  hat.config.test((224, 224, 3), (1000,))
  mod = mobilenetv2()
  mod.summary()
  import tensorflow as tf
  try:
    tf.compat.v1.keras.utils.plot_model(
        mod.model,
        to_file='./unpush/mobilenetv2.png',
        show_shapes=True,)
  except Exception:
    pass

