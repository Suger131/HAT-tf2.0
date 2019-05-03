'''
  WRN-28-10
  本模型默认总参数量[参考基准：cifar10模型]：
  S = 16
  K = 10
  N = 3
    Total params:           36,497,146
    Trainable params:       36,479,194
    Non-trainable params:   17,952
'''

from models.network import NetWork
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import SGD


class wrn28_10(NetWork):
  """
  WRN-28-10 模型
  """

  def args(self):
    # counters
    self._COUNT_CONV = 0
    self._COUNT_BN = 0
    self._COUNT_RELU = 0

    # main args
    self.S = 16
    self.K = 10
    self.N = 3
    self.D = 0

    # train args
    self.EPOCHS = 200
    self.BATCH_SIZE = 32
    self.OPT = SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
    self.OPT_EXIST = True

  def build_model(self):
    x_in = Input(shape=self.IMAGE_SHAPE)
    Fs = [x * self.S * self.K for x in [1, 2, 4]]

    # Conv1
    x = self.conv(x_in, self.S, 3)

    # Res Blocks
    x = self.conv_block(x, Fs[0], self.N, strides=1)
    x = self.conv_block(x, Fs[1], self.N)
    x = self.conv_block(x, Fs[2], self.N)

    # Output
    x = self.bn(x)
    x = self.relu(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(self.NUM_CLASSES, activation='softmax', name='softmax')(x)

    self.model = Model(inputs=x_in, outputs=x, name='wrn28_10')

  def conv(self, x_in, filters, kernel_size, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal'):
    '''卷积层'''
    self._COUNT_CONV += 1
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               kernel_initializer=kernel_initializer,
               name='CONV_' + str(self._COUNT_CONV))(x_in)
    return x

  def bn(self, x_in, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'):
    '''BN层'''
    self._COUNT_BN += 1
    x = BatchNormalization(axis=-1,
                           momentum=momentum,
                           epsilon=epsilon,
                           gamma_initializer=gamma_initializer,
                           name='BN_' + str(self._COUNT_BN))(x_in)
    return x

  def relu(self, x_in):
    '''RELU层'''
    self._COUNT_RELU += 1
    x = Activation('relu', name='RELU_' + str(self._COUNT_RELU))(x_in)
    return x

  def expand(self, x_in, F, strides=2):
    '''塑形层

       要确保使用之前紧接着的是卷积层'''
    xi = self.bn(x_in)
    xi = self.relu(xi)
    
    x = self.conv(xi, F, 3, strides=strides)
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x, F, 3)

    x_aux = self.conv(xi, F, 1, strides=strides)

    x = Add()([x, x_aux])
    return x

  def conv_block(self, x_in, F, N, strides=2):
    '''残差块'''
    
    xi = self.expand(x_in, F, strides=strides)

    for i in range(N):

      x = self.bn(xi)
      x = self.relu(x)
      x = self.conv(x, F, 3)

      x = Dropout(self.D)(x)

      x = self.bn(x)
      x = self.relu(x)
      x = self.conv(x, F, 3)

      xi = Add()([x, xi])

    return xi


# test part
if __name__ == "__main__":
  mod = wrn28_10(DATAINFO={'IMAGE_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.model.summary())
  # pass