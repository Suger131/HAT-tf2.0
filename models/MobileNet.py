'''
  本模型基于Google的MobileNet-v2
  根据cifar10数据集进行一定的修改
  总卷积层数：30
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           741,242
    Trainable params:       724,698
    Non-trainable params:   16,544
'''


from .Model import BasicModel
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import SGD

class MobileNet(BasicModel):

  def __init__(self, i_s, i_d, n_s, Args):
    super(MobileNet, self).__init__(Args)
    self.INPUT_SHAPE = i_s
    self.INPUT_DEPTH = i_d
    self.NUM_CLASSES = n_s

    # sys args
    self.AXIS = K.image_data_format() == 'channels_first' and 1 or - 1

    # counters
    self._COUNT_CONV = 0
    self._COUNT_DWCONV = 0
    
    # Main args
    self.T = 6
    self.D = 0.3

    # train args
    # self.EPOCHS = 200
    # self.BATCH_SIZE = 32
    # self.OPT = SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
    # self.OPT_EXIST = True

    self.create_model()

  def _conv(self, x_in, filters, kernel=1, strides=1, padding='same', use_act=True):
    '''conv with BatchNormalization'''
    self._COUNT_CONV += 1
    x = Conv2D(filters, kernel, strides=strides, padding=padding,
      name= f'CONV_{self._COUNT_CONV}_F{filters}_K{kernel}_S{strides}')(x_in)
    x = BatchNormalization(axis=self.AXIS)(x)
    if use_act: x = Activation('relu')(x)
    return x

  def _DWconv(self, x_in, kernel, strides=1, padding='same'):
    '''DepthwiseConv with BatchNormalization'''
    self._COUNT_DWCONV += 1
    x = DepthwiseConv2D(kernel, strides=strides, depth_multiplier=1, padding=padding,
      name= f'DWCONV_{self._COUNT_DWCONV}_K{kernel}_S{strides}')(x_in)
    x = BatchNormalization(axis=self.AXIS)(x)
    x = Activation('relu')(x)
    return x

  def _bottleneck(self, x_in, filters, kernel, t, s, res=False):
    '''bottleneck'''
    
    tchannel = K.int_shape(x_in)[self.AXIS] * t
    x = self._conv(x_in, tchannel, 1, 1)
    x = self._DWconv(x, kernel, s)
    x = self._conv(x, filters, 1, 1, use_act=False)
    if res: x = Add()([x, x_in])

    return x

  def _inverted_residual_block(self, x_in, filters, kernel, t, n, strides=1):
    '''Inverted Residual Block'''

    x = self._bottleneck(x_in, filters, kernel, t, strides)

    for i in range(n - 1):
      x = self._bottleneck(x, filters, kernel, t, 1, res=True)

    return x

  def create_model(self):
    x_in = Input(shape=(self.INPUT_SHAPE, self.INPUT_SHAPE, self.INPUT_DEPTH))
    
    x = self._conv(x_in, 32, 3, 1)
    x = self._inverted_residual_block(x, 48, 3, 1, 1, 2)
    x = self._inverted_residual_block(x, 64, 3, self.T, 4, 1)
    x = self._inverted_residual_block(x, 96, 3, self.T, 3, 2)
    x = self._inverted_residual_block(x, 160, 3, self.T, 1, 1)
    x = self._conv(x, 320, 1, 1)

    x = GlobalAvgPool2D()(x)
    x = Reshape((1, 1, 320))(x)
    x = Dropout(self.D, name='Dropout')(x)
    # x = Dense(self.NUM_CLASSES, activation='softmax', name='softmax')(x)
    x = Conv2D(self.NUM_CLASSES, 1, padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    x = Reshape((self.NUM_CLASSES,))(x)
    self.model = Model(inputs=x_in, outputs=x, name='MobileNet')

    self.check_save()

# test part
if __name__ == "__main__":
  mod = MobileNet(32, 3, 10, None)
  # print(mod.model.summary())
  print(mod._COUNT_CONV + mod._COUNT_DWCONV + 1)
