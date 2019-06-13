"""
  ZF-Net 模型
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           24,748,554
    Trainable params:       24,747,850
    Non-trainable params:   704
  本模型默认总参数量[参考基准：ImageNet]：
    Total params:           62,359,016
    Trainable params:       62,358,312
    Non-trainable params:   704
"""

# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import

from tensorflow.python.keras.models import Model
from hat.models.network import NetWork
from hat.models.advance import AdvNet


class zfnet(NetWork, AdvNet):
  """
    ZFNet
  """
  def args(self):
    self.CONV = [96, 256, 384, 384, 256]
    self.SIZE = [7, 5, 3, 3, 3]
    self.STEP = [2, 2 if self.INPUT_SHAPE[0] >= 160 else 1, 2, 2, 2]
    self.PAD = 'valid' if self.INPUT_SHAPE[0] >= 160 else 'same'
    self.POOL_SIZE = 3
    self.LOCAL = [4096, 4096]
    self.DROP = 0.5

  def build_model(self):
    
    x_in = self.input(self.INPUT_SHAPE)

    # conv
    x = self.conv(x_in, self.CONV[0], self.SIZE[0], strides=self.STEP[0], padding=self.PAD, activation='relu')
    x = self.bn(x)
    x = self.maxpool(x, self.POOL_SIZE, self.STEP[1])
    x = self.conv(x, self.CONV[1], self.SIZE[1], strides=self.STEP[2], padding=self.PAD, activation='relu')
    x = self.bn(x)
    x = self.maxpool(x, self.POOL_SIZE, self.STEP[3])
    x = self.conv(x, self.CONV[2], self.SIZE[2], activation='relu')
    x = self.conv(x, self.CONV[3], self.SIZE[3], activation='relu')
    x = self.conv(x, self.CONV[4], self.SIZE[4], activation='relu')
    x = self.maxpool(x, self.POOL_SIZE, self.STEP[4], padding=self.PAD)

    # local
    x = self.flatten(x)
    x = self.local(x, self.LOCAL[0])
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.LOCAL[1])
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='zfnet')


# test part
if __name__ == "__main__":
  mod = zfnet(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
