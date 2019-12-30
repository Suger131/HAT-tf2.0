# -*- coding: utf-8 -*-
"""LeNet

  File: 
    /hat/model/test/lenet

  Description: 
    LeNet-5模型
    卷积层深度分别为20和50，卷积核大小为5
    全连接层1节点数为500，随机失活率为0.5
    *基于Network_v2
"""


import hat


# import setting
__all__ = [
  'lenet',
]


class lenet(hat.Network):
  """LeNet

    Description: 
      LeNet-5. Based on <Gradient-Based Learning Applied to 
      Document Recognition>. This version is referenced for 
      the Cifar-10 dataset.
      SEE: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

      *为实验Network_v2的Block功能，临时性添加了Block

    Attributes:
      Consistent with Network.

    Overrided:
      args: 存放参数
      build: 定义了`keras.Model`并返回
  """
  def args(self):
    self.node = 500
    self.drop = 0.5
    self.block1 = self.nn.Block('Graph')
    self.block2 = self.nn.Block('Node')

  def build(self):
    inputs = self.nn.input(self.config.input_shape)
    # block1
    x = self.nn.conv(20, 5, padding='valid', activation='relu', block=self.block1)(inputs)
    x = self.nn.maxpool(3, 2, block=self.block1)(x)
    x = self.nn.conv(50, 5, padding='valid', activation='relu', block=self.block1)(x)
    x = self.nn.maxpool(3, 2, block=self.block1)(x)
    # block2
    x = self.nn.flatten(block=self.block2)(x)
    x = self.nn.dense(self.node, activation='relu', block=self.block2)(x)
    x = self.nn.dropout(self.drop, block=self.block2)(x)
    x = self.nn.dense(self.config.output_shape[-1], activation='softmax', block=self.block2)(x)
    return self.nn.model(inputs, x)


# test
if __name__ == "__main__":
  t = hat._TC()
  t.input_shape = (28, 28, 1)
  t.output_shape = (10,)
  mod = lenet(config=t)
  t.model.summary()
  print(mod.nn.get_block_layer(mod.model, mod.block1))
