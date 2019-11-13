"""
  hat.model.test.mlp
  默认模型
  简单三层神经网络[添加了Dropout层]
  Network v2
"""


from hat.model.utils import Network


class mlp(Network):
  """
    MLP
  """
  def args(self):
    self.node = 128
    self.drop = 0.5
    self.block = self.nn.Block('MLP')

  def build(self):
    inputs = self.nn.input(self.config.input_shape)
    x = self.nn.flatten(block=self.block)(inputs)
    x = self.nn.dense(self.node, block=self.block)(x)
    x = self.nn.dropout(self.drop)(x)
    x = self.nn.dense(self.config.output_shape[-1], activation='softmax')(x)
    return self.nn.model(inputs, x)


# test
if __name__ == "__main__":
  from hat.utils._TC import _TC
  t = _TC()
  t.input_shape = (28, 28, 1)
  t.output_shape = (10,)
  mod = mlp(t)
  t.model.summary()
  print(mod.nn.get_block_layer(mod.model, mod.block))


