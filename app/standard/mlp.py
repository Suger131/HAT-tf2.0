# -*- coding: utf-8 -*-
"""MLP

  File: 
    /hat/app/standard/mlp

  Description: 
    MLP模型
    简单三层神经网络[添加了Dropout层]
    *基于Network_v2
"""


import hat


class mlp(hat.Network):
  """MLP

    Description:
      MLP模型，简单三层神经网络
      添加了Dropout

    Args:
      None

    Overrided:
      args: 存放参数
      build: 定义了`keras.Model`并返回
  """
  def args(self):
    self.node = 128
    self.drop = 0.5

  def build(self):
    inputs = self.nn.input(self.input_shape)
    x = self.nn.flatten()(inputs)
    x = self.nn.dense(self.node)(x)
    x = self.nn.dropout(self.drop)(x)
    x = self.nn.dense(self.output_class, activation='softmax')(x)
    return self.nn.model(inputs, x)


# test
if __name__ == "__main__":
  hat.config.test((28, 28, 1), (10,))
  mod = mlp()
  mod.summary()

