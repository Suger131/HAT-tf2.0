'''
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           437,588
    Trainable params:       437,588
    Non-trainable params:   0
'''


from .Model import BasicModel
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *

class LeNet(BasicModel):

  def __init__(self, i_s, i_d, n_s, Args):
    super(LeNet, self).__init__(Args)
    self.INPUT_SHAPE = i_s
    self.INPUT_DEPTH = i_d
    self.NUM_CLASSES = n_s

    self.S = 5
    self.P = 3
    self.LOCAL_SIZE = 128
    self.DROP_RATE = .5

    self.model = Sequential()
    self.model.add(Conv2D(20, self.S, padding='same', activation='relu',
                          input_shape=(self.INPUT_SHAPE, self.INPUT_SHAPE, self.INPUT_DEPTH)))
    self.model.add(MaxPool2D(self.P, 2, padding='same'))
    self.model.add(Conv2D(50, self.S, padding='same', activation='relu'))
    self.model.add(MaxPool2D(self.P, 2, padding='same'))
    self.model.add(Flatten())
    self.model.add(Dense(self.LOCAL_SIZE, activation='relu'))
    self.model.add(Dropout(self.DROP_RATE))
    self.model.add(Dense(self.NUM_CLASSES, activation='softmax', name='softmax'))

    self.check_save()

# test part
if __name__ == "__main__":
  mod = LeNet(32, 3, 10, None)
  print(mod.model.summary())
