"""
  hat.model.test.lenet

  Network v2
"""


import hat


class lenet(hat.Network):
  """
    LeNet-5
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
  mod = lenet(t)
  t.model.summary()
  print(mod.nn.get_block_layer(mod.model, mod.block1))


