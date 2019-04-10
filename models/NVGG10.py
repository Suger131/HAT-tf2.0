'''
  基于VGG思路构建的快速版本
  该版本为10个卷积层
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           77,130
    Trainable params:       76,554
    Non-trainable params:   576
'''


from .Model import BasicModel
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import SGD


class NVGG10(BasicModel):

  def __init__(self, i_s, i_d, n_s, Args):
    super(NVGG10, self).__init__(Args)
    self.INPUT_SHAPE = i_s
    self.INPUT_DEPTH = i_d
    self.NUM_CLASSES = n_s

    self._COUNT_CONV = 0

    self.F = 32
    self.S = 3
    self.P = 3
    self.ST = 2
    self.LOCAL = 32
    self.DROP_RATE = 0

    # self.BATCH_SIZE = 128
    # self.EPOCHS = 150
    # self.OPT = SGD(lr=0.001, momentum=0.9, decay=5e-4, nesterov=True)
    # self.OPT_EXIST = True

    self.create_model()

  def conv_bn(self, x_in, filters, kernel_size, strides=1, padding='same', use_bias=False):
    '''带有batchnorm层的卷积'''
    self._COUNT_CONV += 1
    with K.name_scope('conv_' + str(self._COUNT_CONV)):
      x = Conv2D(filters,
                 kernel_size,
                 strides=strides,
                 padding=padding,
                 use_bias=use_bias,
                 name='conv_' + str(self._COUNT_CONV))(x_in)
      x = BatchNormalization(name='bn_' + str(self._COUNT_CONV))(x)
      x = Activation('relu', name='relu_' + str(self._COUNT_CONV))(x)
    return x

  def create_model(self):
    x_in = Input(shape=(self.INPUT_SHAPE, self.INPUT_SHAPE, self.INPUT_DEPTH))
    x = self.conv_bn(x_in, self.F, self.S)
    x = MaxPool2D(self.P, self.ST, padding='same')(x)
    for i in range(4):
      x = self.conv_bn(x, self.F, self.S)
    x = MaxPool2D(self.P, self.ST, padding='same')(x)
    for i in range(4):
      x = self.conv_bn(x, self.F, self.S)
    x = GlobalAvgPool2D()(x)
    x = Dense(self.LOCAL, activation='relu')(x)
    x = Dropout(self.DROP_RATE)(x)
    x = Dense(self.NUM_CLASSES, activation='softmax')(x)
    self.model = Model(inputs=x_in, outputs=x, name='NVGG10')
    self.check_save()

# test part
if __name__ == "__main__":
  mod = NVGG10(32, 3, 10, None)
  print(mod.model.summary())