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

    self.F1 = 20
    self.S1 = 5
    self.P1 = 3
    self.PS1 = 2
    self.F2 = 50
    self.S2 = 5
    self.P2 = 3
    self.PS2 = 2
    self.LOCAL_SIZE = 128
    self.DROP_RATE = .5

    self.model = Sequential()
    self.model.add(Conv2D(self.F1, self.S1, padding='same', activation='relu',
                          input_shape=(self.INPUT_SHAPE, self.INPUT_SHAPE, self.INPUT_DEPTH)))
    self.model.add(MaxPool2D(self.P1, self.PS1, padding='same'))
    self.model.add(Conv2D(self.F2, self.S2, padding='same', activation='relu'))
    self.model.add(MaxPool2D(self.P2, self.PS2, padding='same'))
    self.model.add(Flatten())
    self.model.add(Dense(self.LOCAL_SIZE, activation='relu'))
    self.model.add(Dropout(self.DROP_RATE))
    self.model.add(Dense(self.NUM_CLASSES, activation='softmax'))

    self.check_save()

# test part
if __name__ == "__main__":
  mod = MLP(32, 3, 10, None)
  print(mod.model.summary())
