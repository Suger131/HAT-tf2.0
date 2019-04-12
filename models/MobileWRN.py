'''
  本模型结合Google的MobileNet-v2和WRN
  根据cifar10数据集进行一定的修改
  MobileWRN-38-2-8
  总卷积层数：
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           679,626
    Trainable params:       665,930
    Non-trainable params:   13,696
'''


from .Model import BasicModel
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import SGD

class MobileWRN(BasicModel):

  def __init__(self, i_s, i_d, n_s, Args):
    super(MobileWRN, self).__init__(Args)
    self.INPUT_SHAPE = i_s
    self.INPUT_DEPTH = i_d
    self.NUM_CLASSES = n_s

    # sys args
    self.AXIS = K.image_data_format() == 'channels_first' and 1 or - 1

    # counters
    self._COUNT_CONV = 0
    self._COUNT_DWCONV = 0
    
    # Main args
    self.S = 16
    self.W = 2
    self.K = 6
    self.N = 3
    self.D = 0.2
    self.F = [x * self.S * self.W for x in [1, 2, 4]]
    self.L = self.F[-1]

    # train args
    # self.EPOCHS = 200
    # self.BATCH_SIZE = 32
    # self.OPT = SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
    # self.OPT_EXIST = True

    self.create_model()

  def _conv(self, x_in, filters, kernel=1, strides=1, padding='same', use_bn=True, use_act=True):
    '''conv with BatchNormalization and ReLU'''
    self._COUNT_CONV += 1
    x = Conv2D(filters, kernel, strides=strides, padding=padding,
      name= f'CONV_{self._COUNT_CONV}_F{filters}_K{kernel}_S{strides}')(x_in)
    if use_bn: x = BatchNormalization(axis=self.AXIS)(x)
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

  def _DW_block(self, x_in, k, filters, kernel, strides, res=False):
    '''使用DWconv的网络块，残差可选'''

    kchannel = K.int_shape(x_in)[self.AXIS] * k
    x = BatchNormalization(axis=self.AXIS)(x_in)
    x = Activation('relu')(x)
    x = self._conv(x_in, kchannel, 1, 1)
    x = self._DWconv(x, kernel, strides)
    x = self._conv(x, filters, 1, 1, use_bn=False, use_act=False)
    if res: x = Add()([x, x_in])

    return x

  def _n_block(self, x_in, n, k, filters, kernel, strides=1):
    '''Inverted Residual Block'''

    x = self._DW_block(x_in, k, filters, kernel, strides)

    for i in range(n - 1):
      x = self._DW_block(x, k, filters, kernel, 1, res=True)

    return x

  def _Output(self, x_in):
    '''Softmax Output

      Use Conv instead of Dense'''
    x = BatchNormalization()(x_in)
    x = Activation('relu')(x)
    x = GlobalAvgPool2D()(x)
    x = Reshape((1, 1, self.L))(x)
    x = Dropout(self.D, name='Dropout')(x)
    x = Conv2D(self.NUM_CLASSES, 1, padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    x = Reshape((self.NUM_CLASSES,))(x)
    return x

  def create_model(self):
    x_in = Input(shape=(self.INPUT_SHAPE, self.INPUT_SHAPE, self.INPUT_DEPTH))
    
    x = self._conv(x_in, self.S, 3, 1, use_bn=False, use_act=False)

    x = self._n_block(x, self.N, self.K, self.F[0], 3, 1)
    x = self._n_block(x, self.N, self.K, self.F[1], 3, 2)
    x = self._n_block(x, self.N, self.K, self.F[2], 3, 2)

    x = self._Output(x)
    
    self.model = Model(inputs=x_in, outputs=x, name='MobileWRN')

    self.check_save()

# test part
if __name__ == "__main__":
  mod = MobileWRN(32, 3, 10, None)
  print(mod.model.summary())
  print(mod._COUNT_CONV + mod._COUNT_DWCONV + 1)
