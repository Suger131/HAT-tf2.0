"""
  LeNet-GroupConv
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           1,633,204
    Trainable params:       1,633,204
    Non-trainable params:   0
  注：
    1、比LeNet标准模型只增加了1124参数量
    2、原本的MaxPool改成了MixPool
"""


from hat.models.advance import AdvNet


class lg(AdvNet):
  """
    LeNet-GroupConv
  """
  def args(self):
    self.G = 5
    self.CONV = [20, 50]
    self.SIZE = 5
    self.LOCAL = 500
    self.DROP = 0.5
  
  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    x = x_in
    x = self.conv(x, self.CONV[0], self.SIZE, activation='relu')
    x = self.Gconv(x, self.G, 0, 3, use_group_bias=True)
    x = self.add([self.maxpool(x), self.avgpool(x)])
    x = self.conv(x, self.CONV[1], self.SIZE, activation='relu')
    x = self.Gconv(x, self.G, 0, 3, use_group_bias=True)
    x = self.add([self.maxpool(x), self.avgpool(x)])

    x = self.flatten(x)
    x = self.local(x, self.LOCAL)
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='lg')

# test part
if __name__ == "__main__":
  mod = lg(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10}, built=True)
  mod.summary()
