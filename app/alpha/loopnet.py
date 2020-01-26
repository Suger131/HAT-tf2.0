# -*- coding: utf-8 -*-
"""Loop

  File: 
    /hat/app/alpha/loop

  Description: 
    Loop模型
    *基于Network_v2
"""


import hat


class loopnet1(hat.Network):
  """LoopNet1

    Descrtption:
      Loop模型，基于VGG16
      将原本的重复层替换为Loop
  """
  def args(self):
    self.drop = .2
    self.ratio = 1
    hat.config.set('opt', hat.nn.optimizers.Adam(lr=1e-3))

  def build(self):
    inputs = self.nn.input(self.input_shape)
    x = inputs
    x = self.nn.loopconv2d(1, 64, 3, ratio=self.ratio, activation='relu')(x)
    x = self.nn.maxpool(3, 2)(x)
    x = self.nn.loopconv2d(1, 128, 3, ratio=self.ratio, activation='relu')(x)
    x = self.nn.maxpool(3, 2)(x)
    x = self.nn.loopconv2d(2, 256, 3, ratio=self.ratio, activation='relu')(x)
    x = self.nn.maxpool(3, 2)(x)
    x = self.nn.loopconv2d(2, 512, 3, ratio=self.ratio, activation='relu')(x)
    x = self.nn.maxpool(3, 2)(x)
    x = self.nn.loopconv2d(2, 512, 3, ratio=self.ratio, activation='relu')(x)
    x = self.nn.maxpool(3, 2)(x)
    x = self.nn.flatten()(x)
    x = self.nn.loopdense(1, 4096, ratio=self.ratio, activation='relu')(x)
    x = self.nn.dropout(self.drop)(x)
    x = self.nn.dense(self.output_class, activation='softmax')(x)
    return self.nn.model(inputs, x)


# test
if __name__ == "__main__":
  hat.config.test((32, 32, 3), (10,))
  mod = loopnet1()
  mod.summary()

