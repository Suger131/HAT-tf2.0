'''
  经典的VGG10网络
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           9,517,098
    Trainable params:       9,517,098
    Non-trainable params:   0
'''


from .Model import BasicModel
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import SGD


class VGG10(BasicModel):

  def __init__(self, i_s, i_d, n_s, Args):
    super(VGG10, self).__init__(Args)
    self.INPUT_SHAPE = i_s
    self.INPUT_DEPTH = i_d
    self.NUM_CLASSES = n_s

    self.FA = 32
    self.FB = 64
    self.FC = 128
    self.S = 3
    self.P = 3
    self.ST = 2
    self.LOCAL_A = 512
    self.LOCAL_B = 128
    self.LOCAL_C = 128
    self.DROP_RATE = 0

    # self.BATCH_SIZE = 128
    # self.EPOCHS = 150
    # self.OPT = SGD(lr=0.001, momentum=0.9, decay=5e-4, nesterov=True)
    # self.OPT_EXIST = True

    self.model = Sequential([
      Conv2D(self.FA, self.S, padding='same', activation='relu', input_shape=(i_s, i_s, i_d)),
      Conv2D(self.FA, self.S, padding='same', activation='relu'),
      MaxPool2D(self.P, self.ST, padding='same'),
      Conv2D(self.FB, self.S, padding='same', activation='relu'),
      Conv2D(self.FB, self.S, padding='same', activation='relu'),
      MaxPool2D(self.P, self.ST, padding='same'),
      Conv2D(self.FC, self.S, padding='same', activation='relu'),
      Conv2D(self.FC, self.S, padding='same', activation='relu'),
      Conv2D(self.FC, self.S, padding='same', activation='relu'),
      GlobalMaxPool2D(),
      Flatten(),
      Dense(self.LOCAL_A, activation='relu'),
      Dropout(self.DROP_RATE),
      Dense(self.LOCAL_B, activation='relu'),
      Dropout(self.DROP_RATE),
      Dense(self.LOCAL_C, activation='relu'),
      Dropout(self.DROP_RATE),
      Dense(n_s, activation='softmax')
    ])
    self.check_save()

# test part
if __name__ == "__main__":
  mod = VGG10(32, 3, 10, None)
  print(mod.model.summary())