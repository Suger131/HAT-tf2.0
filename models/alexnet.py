'''
    AlexNet 模型
    For ImageNet
    本模型默认总参数量[参考基准：ImageNet]：
    Total params:           889,351,682
    Trainable params:       889,350,978
    Non-trainable params:   704
'''


from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras.models import Model


class alexnet(NetWork, AdvNet):
  """
    AlexNet
  """
  def args(self):
    self.CONV = [48, 128, 192, 192, 128]
    self.SIZE = [11, 27, 13, 13, 13]
    self.POOL_SIZE = 3
    self.POOL_STRIDES = 2
    self.LOCAL = [2048, 2048, 1000]
    self.DROP = 0.5

  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    # conv
    x1 = self.conv(x_in, self.CONV[0], self.SIZE[0], activation='relu')
    x2 = self.conv(x_in, self.CONV[0], self.SIZE[0], activation='relu')
    x1 = self.maxpool(x1, self.POOL_SIZE, self.POOL_STRIDES)
    x2 = self.maxpool(x2, self.POOL_SIZE, self.POOL_STRIDES)
    x1 = self.bn(x1)
    x2 = self.bn(x2)
    x1 = self.conv(x1, self.CONV[1], self.SIZE[1], activation='relu')
    x2 = self.conv(x2, self.CONV[1], self.SIZE[1], activation='relu')
    x1 = self.maxpool(x1, self.POOL_SIZE, self.POOL_STRIDES)
    x2 = self.maxpool(x2, self.POOL_SIZE, self.POOL_STRIDES)
    x1 = self.bn(x1)
    x2 = self.bn(x2)
    x = self.concat([x1, x2])
    x1 = self.conv(x, self.CONV[2], self.SIZE[2], activation='relu')
    x2 = self.conv(x, self.CONV[2], self.SIZE[2], activation='relu')
    x1 = self.conv(x1, self.CONV[3], self.SIZE[3], activation='relu')
    x2 = self.conv(x2, self.CONV[3], self.SIZE[3], activation='relu')
    x1 = self.conv(x1, self.CONV[4], self.SIZE[4], activation='relu')
    x2 = self.conv(x2, self.CONV[4], self.SIZE[4], activation='relu')
    x1 = self.maxpool(x1, self.POOL_SIZE, self.POOL_STRIDES)
    x2 = self.maxpool(x2, self.POOL_SIZE, self.POOL_STRIDES)
    x = self.concat([x1, x2])

    # local
    x = self.flatten(x)
    x1 = self.local(x, self.LOCAL[0])
    x2 = self.local(x, self.LOCAL[0])
    x = self.concat([x1, x2])
    x = self.dropout(x, self.DROP)
    x1 = self.local(x, self.LOCAL[1])
    x2 = self.local(x, self.LOCAL[1])
    x = self.concat([x1, x2])
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.LOCAL[2])
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='alexnet')


# test part
if __name__ == "__main__":
  mod = alexnet(DATAINFO={'INPUT_SHAPE': (224, 224 ,3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
