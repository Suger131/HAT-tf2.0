'''
  AlexNet-s 模型
  For Cifar10
  本模型默认总参数量[参考基准：ImageNet]：
    Total params:           889,351,682
    Trainable params:       889,350,978
    Non-trainable params:   704
'''


from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras.models import Model


class alexnet_s(NetWork, AdvNet):
  """
    AlexNet-s
  """
  def args(self):
    self.CONV = [32, 64, 96, 128, 256]
    self.TIME = [2, 4, 3, 3, 3]
    self.POOL_SIZE = 3
    self.POOL_STRIDES = 2
    self.LOCAL = 1024
    self.DROP = 0.5

  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    # conv
    # part 1
    x1 = self.repeat(self.conv_bn, self.TIME[0], x_in, self.CONV[0], 3)
    x2 = self.repeat(self.conv_bn, self.TIME[0], x_in, self.CONV[0], 3)
    x = self.concat([x1, x2])
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES)
    # part 2
    x1 = self.repeat(self.conv_bn, self.TIME[1], x, self.CONV[1], 3)
    x2 = self.repeat(self.conv_bn, self.TIME[1], x, self.CONV[1], 3)
    x = self.concat([x1, x2])
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES)
    # part 3
    x1 = self.repeat(self.conv_bn, self.TIME[2], x, self.CONV[2], 3)
    x2 = self.repeat(self.conv_bn, self.TIME[2], x, self.CONV[2], 3)
    x = self.concat([x1, x2])
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES)
    # part 3
    x1 = self.repeat(self.conv_bn, self.TIME[3], x, self.CONV[3], 3)
    x2 = self.repeat(self.conv_bn, self.TIME[3], x, self.CONV[3], 3)
    x = self.concat([x1, x2])
    x1 = self.repeat(self.conv_bn, self.TIME[4], x, self.CONV[4], 3)
    x2 = self.repeat(self.conv_bn, self.TIME[4], x, self.CONV[4], 3)
    x = self.concat([x1, x2])

    # local
    x = self.GAPool(x)
    x1 = self.local(x, self.LOCAL)
    x2 = self.local(x, self.LOCAL)
    x3 = self.local(x, self.LOCAL)
    x4 = self.local(x, self.LOCAL)
    x = self.concat([x1, x2, x3, x4])
    x = self.dropout(x, self.DROP)
    x1 = self.local(x, self.LOCAL)
    x2 = self.local(x, self.LOCAL)
    x = self.concat([x1, x2])
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.LOCAL)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='alexnet_s')


# test part
if __name__ == "__main__":
  mod = alexnet_s(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
