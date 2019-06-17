'''
  LeNet-Swish-Group
  For cifar10
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           1,800,222
    Trainable params:       1,800,222
    Non-trainable params:   0
'''

# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import

from tensorflow.python.keras.models import Model
from hat.models.advance import AdvNet
from hat.models.network import NetWork


class lsg(NetWork, AdvNet):
  """
    LSG
  """
  def args(self):
    pass

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
    x = self.groupconv(x, 32, 4, 3)
    x = self.bn(x)
    x = self.swish(x)

    # Down sampling
    x = self.add([self.maxpool(x), self.avgpool(x)])

    # Stage 2
    x = self.conv(x, 256, 3)
    x = self.bn(x)
    x = self.swish(x)
    x = self.groupconv(x, 32, 8, 3)
    x = self.bn(x)
    x = self.swish(x)

    # Down sampling
    x = self.add([self.maxpool(x), self.avgpool(x)])

    # Stage 3
    x = self.conv(x, 512, 3)
    x = self.bn(x)
    x = self.swish(x)
    x = self.groupconv(x, 32, 16, 3)
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

    self.model = Model(inputs=x_in, outputs=x, name='lsg')


# test part
if __name__ == "__main__":
  mod = lsg(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
