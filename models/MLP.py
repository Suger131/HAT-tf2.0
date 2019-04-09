'''
  默认模型
  简单三层神经网络[添加了Dropout层]
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           394,634
    Trainable params:       394,634
    Non-trainable params:   0
'''

from .Model import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *


class MLP(Model):

  def __init__(self, i_s, i_d, n_s, Args):
    super(MLP, self).__init__(Args)
    self.INPUT_SHAPE = i_s
    self.INPUT_DEPTH = i_d
    self.NUM_CLASSES = n_s
    self.LOCAL_SIZE = 128
    self.DROP_RATE = 0.5

    # for test
    # self.BATCH_SIZE = 128
    # self.EPOCHS = 150
    # self.OPT = 'sgd'
    # self.OPT_EXIST = True

    self.model = self.model or Sequential([
      Flatten(input_shape=(i_s, i_s, i_d)),
      Dense(self.LOCAL_SIZE, activation='relu'),
      Dropout(self.DROP_RATE),
      Dense(n_s, activation='softmax')
    ])

# test part
if __name__ == "__main__":
  mod = MLP(32, 3, 10, None)
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
