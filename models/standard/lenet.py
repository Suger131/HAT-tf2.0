"""
  LeNet 模型
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           1,632,080
    Trainable params:       1,632,080
    Non-trainable params:   0
"""

# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import

from tensorflow.python.keras.models import Model
from hat.models.advance import AdvNet


class lenet(AdvNet):
  """
    LeNet 
  """

  def args(self):
    self.CONV = [20, 50]
    self.SIZE = 5
    self.POOL_SIZE = 3
    self.POOL_STEP = 2
    self.LOCAL = 500
    self.DROP = 0.5
    
  def build_model(self):

    x_in = self.input(self.INPUT_SHAPE)

    # conv
    x = x_in
    x = self.conv(x, self.CONV[0], self.SIZE, activation='relu')
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STEP)
    x = self.conv(x, self.CONV[1], self.SIZE, activation='relu')
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STEP)

    # local
    x = self.flatten(x)
    x = self.local(x, self.LOCAL)
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='lenet')


# test part
if __name__ == "__main__":
  mod = lenet(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
