'''
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           1,056,410
    Trainable params:       1,056,410
    Non-trainable params:   0
'''


from .Model import BasicModel
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model


class David(BasicModel):

  def __init__(self, i_s, i_d, n_s, Args):
    super(David, self).__init__(Args)
    self.INPUT_SHAPE = i_s
    self.INPUT_DEPTH = i_d
    self.NUM_CLASSES = n_s

    # counters
    self._COUNT_CONV = 0
    self._COUNT_BN = 0
    self._COUNT_RELU = 0

    # main args
    self.S = 16
    self.K = 10
    self.N = 3
    self.D = 0

    self.S = 3
    self.T = 2
    self.F = [32, 48, 80, 128]
    self.DROP = 0.5
    self.LOCAL = 400

    # train args
    # self.EPOCHS = 200
    # self.BATCH_SIZE = 32
    # self.OPT = SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
    # self.OPT_EXIST = True

    self.create_model()

  def conv(self, x_in, filters, kernel_size, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', activation='relu'):
    '''卷积层'''
    self._COUNT_CONV += 1
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               kernel_initializer=kernel_initializer,
               activation=activation,
               name='CONV_' + str(self._COUNT_CONV))(x_in)
    return x

  def create_model(self):
    x_in = Input(shape=(self.INPUT_SHAPE, self.INPUT_SHAPE, self.INPUT_DEPTH))

    x = x_in
    for i in range(3):
      x = self.conv(x, self.F[0], self.S)
    for i in range(2):
      x = self.conv(x, self.F[1], self.S)
    x = MaxPool2D(self.S, self.T, padding='same')(x)
    x = Dropout(self.DROP)(x)

    for i in range(5):
      x = self.conv(x, self.F[2], self.S)
    x = MaxPool2D(self.S, self.T, padding='same')(x)
    x = Dropout(self.DROP)(x)

    for i in range(5):
      x = self.conv(x, self.F[3], self.S)
    x = GlobalMaxPool2D()(x)
    x = Dropout(self.DROP)(x)

    x = Dense(self.LOCAL, activation='relu')(x)
    x = Dropout(self.DROP)(x)
    x = Dense(self.NUM_CLASSES, activation='softmax', name='softmax')(x)

    self.model = Model(inputs=x_in, outputs=x, name='David')
    self.check_save()

# test part
if __name__ == "__main__":
  mod = David(32, 3, 10, None)
  print(mod.model.summary())



