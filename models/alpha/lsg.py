'''
  LeNet-Swish-Group
  For cifar10
  Groups: 16
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           1,840,970
    Trainable params:       1,836,362
    Non-trainable params:   4,608
  NOTE: 对比普通卷积:
    Total params:           4,925,450
    Trainable params:       4,920,842
    Non-trainable params:   4,608
'''


from hat.models.advance import AdvNet


class lsg(AdvNet):
  """
    LSG
  """
  def args(self):
    self.G = 16

  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    x = x_in
    
    x = self.conv(x, 64, 3)
    x = self.swish(x)
    # x = self.conv(x, 64, 3)
    # x = self.swish(x)

    # Down sampling
    x = self.add([self.maxpool(x), self.avgpool(x)])

    # Stage 1
    x = self.conv(x, 128, 3)
    x = self.bn(x)
    x = self.swish(x)
    x = self.Gconv(x, self.G, 0, 3)
    x = self.bn(x)
    x = self.swish(x)

    # Down sampling
    x = self.add([self.maxpool(x), self.avgpool(x)])

    # Stage 2
    x = self.conv(x, 256, 3)
    x = self.bn(x)
    x = self.swish(x)
    x = self.Gconv(x, self.G, 0, 3)
    x = self.bn(x)
    x = self.swish(x)

    # Down sampling
    x = self.add([self.maxpool(x), self.avgpool(x)])

    # Stage 3
    x = self.conv(x, 512, 3)
    x = self.bn(x)
    x = self.swish(x)
    x = self.Gconv(x, self.G, 0, 3)
    x = self.bn(x)
    x = self.swish(x)

    # Global Pooling
    x = self.add([self.GAPool(x), self.GMPool(x)])
    # x = self.GAPool(x)

    x = self.local(x, 512, activation=None)
    x = self.bn(x)
    x = self.swish(x)
    x = self.dropout(x, 0.5)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.Model(inputs=x_in, outputs=x, name='lsg')


# test part
if __name__ == "__main__":
  mod = lsg(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
