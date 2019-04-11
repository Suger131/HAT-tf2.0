'''
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           10,782,314
    Trainable params:       10,749,802
    Non-trainable params:   32,512
'''

from .Model import BasicModel
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import SGD

class GoogleNet(BasicModel):

  def __init__(self, i_s, i_d, n_s, Args):
    super(GoogleNet, self).__init__(Args)
    self.INPUT_SHAPE = i_s
    self.INPUT_DEPTH = i_d
    self.NUM_CLASSES = n_s

    # counters
    self._COUNT_CONV = 0
    self._COUNT_BN = 0
    self._COUNT_RELU = 0
    self._COUNT_REDUCE = 0
    self._COUNT_IA = 0
    self._COUNT_IB = 0
    self._COUNT_IC = 0
    self._COUNT_AUX = 0
    
    # Main args
    self.DROP = 0.2
    self.K = 1
    self.S = 16 * self.K

    # Stem args
    self.STEM_F1 = self.S * 2
    self.STEM_F2 = self.S * 4
    self.STEM_F3 = self.S * 6
    self.STEM_N = 3

    # train args
    self.EPOCHS = 200
    self.BATCH_SIZE = 32
    self.OPT = SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
    self.OPT_EXIST = True

    self.create_model()

  def conv(self, x_in, filters, kernel_size, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal'):
    '''卷积层'''
    self._COUNT_CONV += 1
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               kernel_initializer=kernel_initializer,
               name=f"CONV_{self._COUNT_CONV}_F{filters}_K{type(kernel_size)==int and kernel_size or '%sx%s' % kernel_size}_S{strides}")(x_in)
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

  def conv_bn(self, x_in, filters, kernel_size, strides=1, padding='same', use_bias=False,
              kernel_initializer='he_normal', momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'):
    '''带有bn层的conv'''
    x = self.conv(x_in, filters, kernel_size, strides, padding, use_bias, kernel_initializer)
    x = self.bn(x, momentum, epsilon, gamma_initializer)
    x = self.relu(x)
    return x

  def Reduce(self, x_in, conv_f, size=3, strides=2, padding='same'):
    '''池化和卷积并行的尺寸减少层，默认池化层数等于输入层数'''
    self._COUNT_REDUCE += 1
    pool = MaxPool2D(size, strides=strides, padding=padding, name='Reduce_Maxpool_' + str(self._COUNT_REDUCE))(x_in)
    conv = self.conv_bn(x_in, conv_f, kernel_size=size, strides=strides, padding=padding)
    x = Concatenate()([pool, conv])
    return x

  def InceptionA(self, x_in):
    ''' channels: 4D list
        0: avg
        1: 1x1
        2: 3x3
        3: 3x3 + 3x3'''
    self._COUNT_IA += 1
    with K.name_scope(f'InceptionA_{self._COUNT_IA}'):
      x1 = AvgPool2D(3, strides=1, padding='same')(x_in)
      x1 = self.conv_bn(x1, self.S * 3, 1)

      x2 = self.conv_bn(x_in, self.S * 3, 1)

      x3 = self.conv_bn(x_in, self.S * 2, 1)
      x3 = self.conv_bn(x3, self.S * 3, 3)
      
      x4 = self.conv_bn(x_in, self.S * 2, 1)
      x4 = self.conv_bn(x4, self.S * 3, 3)
      x4 = self.conv_bn(x4, self.S * 3, 3)

      x = Concatenate()([x1, x2, x3, x4])
    return x

  def InceptionB(self, x_in):
    ''' channels: 4D list
        0: avg
        1: 1x1
        2: 3x1 + 1x3
        3: 3x1 + 1x3 + 3x1 + 1x3'''
    self._COUNT_IB += 1
    with K.name_scope(f'InceptionB_{self._COUNT_IB}'):
      x1 = AvgPool2D(3, strides=1, padding='same')(x_in)
      x1 = self.conv_bn(x1, self.S * 4, 1)

      x2 = self.conv_bn(x_in, self.S * 12, 1)

      x3 = self.conv_bn(x_in, self.S * 6, 1)
      x3 = self.conv_bn(x3, self.S * 7, (7, 1))
      x3 = self.conv_bn(x3, self.S * 8, (1, 7))
      
      x4 = self.conv_bn(x_in, self.S * 6, 1)
      x4 = self.conv_bn(x4, self.S * 7, (1, 7))
      x4 = self.conv_bn(x4, self.S * 7, (7, 1))
      x4 = self.conv_bn(x4, self.S * 8, (1, 7))
      x4 = self.conv_bn(x4, self.S * 8, (7, 1))
      x = Concatenate()([x1, x2, x3, x4])
    return x

  def InceptionC(self, x_in):
    ''' channels: 4D list
        0: avg
        1: 1x1
        2: 3x1
        3: 1x3
        4: 3x3 + 3x1
        5: 3x3 + 1x3'''
    self._COUNT_IC += 1
    with K.name_scope(f'InceptionC_{self._COUNT_IC}'):
      x1 = AvgPool2D(3, strides=1, padding='same')(x_in)
      x1 = self.conv_bn(x1, self.S * 8, 1)

      x2 = self.conv_bn(x_in, self.S * 8, 1)

      xa = self.conv_bn(x_in, self.S * 12, 1)
      x3 = self.conv_bn(xa, self.S * 8, (3, 1))
      x4 = self.conv_bn(xa, self.S * 8, (1, 3))
      
      xb = self.conv_bn(x_in, self.S * 12, 1)
      xb = self.conv_bn(xb, self.S * 14, (1, 3))
      xb = self.conv_bn(xb, self.S * 16, (3, 1))
      x5 = self.conv_bn(xb, self.S * 8, (3, 1))
      x6 = self.conv_bn(xb, self.S * 8, (1, 3))
      x = Concatenate()([x1, x2, x3, x4, x5, x6])
    return x

  def Auxout(self, x_in):
    self._COUNT_AUX += 1
    with K.name_scope(f'AUXOUT_{self._COUNT_AUX}'):
      x = GlobalAvgPool2D()(x_in)
      x = Dropout(self.DROP)(x)
      x = Dense(self.NUM_CLASSES, activation='softmax', name=f'Aux_{self._COUNT_AUX}_softmax')(x)
    return x

  def Stem(self, x_in):
    '''Stem'''
    with K.name_scope('Stem'):
      x = self.conv_bn(x_in, self.STEM_F1, 3)
      x = self.conv_bn(x, self.STEM_F1, 3)
      x = self.conv_bn(x, self.STEM_F2, 3)
      x = self.Reduce(x, self.STEM_F3, strides=1)

      x1 = self.conv_bn(x, self.STEM_F2, 1)
      x1 = self.conv_bn(x1, self.STEM_F2, (self.STEM_N, 1))
      x1 = self.conv_bn(x1, self.STEM_F2, (1, self.STEM_N))
      x1 = self.conv_bn(x1, self.STEM_F3, 3)
      
      x2 = self.conv_bn(x, self.STEM_F2, 1)
      x2 = self.conv_bn(x2, self.STEM_F3, 3)

      x = concatenate([x1, x2])
    return x

  def create_model(self):
    x_in = Input(shape=(self.INPUT_SHAPE, self.INPUT_SHAPE, self.INPUT_DEPTH))
    
    # Stem
    x = self.Stem(x_in)

    for i in range(4):
      x = self.InceptionA(x)
    # x_aux1 = self.Auxout(x)

    # Reduce 32x32 -> 16x16, 192+192+128=512
    # x = self.Reduce(x, self.R1)
    x1 = MaxPool2D(3, 2, 'same', name='Reduce_Maxpool_2')(x)
    x2 = self.conv_bn(x, self.S * 12, 3, 2)
    x3 = self.conv_bn(x, self.S * 6, 1)
    x3 = self.conv_bn(x3, self.S * 7, 3)
    x3 = self.conv_bn(x3, self.S * 8, 3, 2)
    x = Concatenate()([x1, x2, x3])

    for i in range(7):
      x = self.InceptionB(x)
    # x_aux2 = self.Auxout(x)
    
    # Reduce 16x16 -> 8x8, 512+96+160=768
    # x = self.Reduce(x, self.R2)
    x1 = MaxPool2D(3, 2, 'same', name='Reduce_Maxpool_3')(x)
    x2 = self.conv_bn(x, self.S * 6, 1)
    x2 = self.conv_bn(x2, self.S * 6, 3, 2)
    x3 = self.conv_bn(x, self.S * 8, 1)
    x3 = self.conv_bn(x3, self.S * 8, (1, 7))
    x3 = self.conv_bn(x3, self.S * 10, (7, 1))
    x3 = self.conv_bn(x3, self.S * 10, 3, 2)
    x = Concatenate()([x1, x2, x3])

    for i in range(3):
      x = self.InceptionC(x)
    x = GlobalAvgPool2D()(x)
    x = Dropout(self.DROP)(x)
    x = Dense(self.NUM_CLASSES, activation='softmax', name='softmax')(x)

    # x = Add()([x_aux1, x_aux2, x])
    self.model = Model(inputs=x_in, outputs=x, name='GoogleNet')

    self.check_save()

# test part
if __name__ == "__main__":
  mod = GoogleNet(32, 3, 10, None)
  print(mod.model.summary())
