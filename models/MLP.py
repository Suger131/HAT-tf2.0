'''
  默认模型
  简单三层神经网络[添加了Dropout层]
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           394,634
    Trainable params:       394,634
    Non-trainable params:   0
'''

from .Model import BasicModel
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *


class MLP(BasicModel):

  def __init__(self, input_shape, num_classes):
    super(MLP, self).__init__()
    self.INPUT_SHAPE = input_shape
    self.NUM_CLASSES = num_classes
    self.LOCAL_SIZE = 128
    self.DROP_RATE = 0.5

    # for test
    # self.BATCH_SIZE = 128
    # self.EPOCHS = 150
    # self.OPT = 'sgd'
    # self.OPT_EXIST = True

    self.model = Sequential([
      Flatten(input_shape=self.INPUT_SHAPE),
      Dense(self.LOCAL_SIZE, activation='relu'),
      Dropout(self.DROP_RATE),
      Dense(self.NUM_CLASSES, activation='softmax')
    ])

# test part
if __name__ == "__main__":
  mod = MLP((32, 32, 3), 10)
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
