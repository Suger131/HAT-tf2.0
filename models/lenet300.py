"""
  LeNet300 模型
  Lite
  本模型默认总参数量[参考基准：car10]：
    Total params:           506,842
    Trainable params:       506,842
    Non-trainable params:   0
"""


from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras.models import Model


class lenet300(NetWork, AdvNet):
  '''
    LeNet300 
  '''
  def args(self):
    self.CONV = [24, 32, 48, 64, 128, 192]
    self.SIZE = 3
    self.POOL_SIZE = 3
    self.POOL_STRIDES = 2
    self.LOCAL = 800
    self.DROP = 0
    
  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    # conv
    x = self.conv(x_in, self.CONV[0], self.SIZE, activation='relu')
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES)
    x = self.conv(x, self.CONV[1], self.SIZE, padding='valid', activation='relu')
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES, padding='valid')
    x = self.conv(x, self.CONV[2], self.SIZE, padding='valid', activation='relu')
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES, padding='valid')
    x = self.conv(x, self.CONV[3], self.SIZE, activation='relu')
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES, padding='valid')
    x = self.conv(x, self.CONV[4], self.SIZE, activation='relu')
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES, padding='valid')
    x = self.conv(x, self.CONV[5], self.SIZE, activation='relu')

    # local
    x = self.GAPool(x)
    x = self.local(x, self.LOCAL)
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='lenet300')


# test part
if __name__ == "__main__":
  mod = lenet300(DATAINFO={'INPUT_SHAPE': (300, 300, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
