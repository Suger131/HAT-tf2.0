"""
  hat.model.test.mlp
  默认模型
  简单三层神经网络[添加了Dropout层]
  Network v2
"""


import hat


class mlp(hat.Network):
  """
    MLP
  """
  def args(self):
    self.node = 128
    self.drop = 0.5
    self.block = self.nn.Block('MLP')

  def build(self):
    inputs = self.nn.input(self.input_shape)
    x = self.nn.flatten(block=self.block)(inputs)
    x = self.nn.dense(self.node, block=self.block)(x)
    x = self.nn.dropout(self.drop)(x)
    x = self.nn.dense(self.output_class, activation='softmax')(x)
    return self.nn.model(inputs, x)


# test
if __name__ == "__main__":
  t = hat.util.Tc()
  t.data.input_shape = (28, 28, 1)
  t.data.output_shape = (10,)
  mod = mlp(config=t)
  mod.summary()
  print(mod.nn.get_layer_with_block(mod.model, mod.block))

