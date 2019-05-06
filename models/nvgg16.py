'''
  NVGG16 模型
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           598,474
    Trainable params:       596,554
    Non-trainable params:   1,920
  本模型默认总参数量[参考基准：ImageNet]：
    Total params:           3,744,202
    Trainable params:       3,742,282
    Non-trainable params:   1,920
'''

from models.network import NetWork
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *


class nvgg16(NetWork):
  """
  NVGG16 模型
  """
  
  def args(self):
    self._COUNT_CONV = 0

    self.NUM_LAYERS = [3, 3, 3, 3, 3]
    self.CONV = 64

    self.LOCAL = 1024
    self.DROP = 0.2

    # for test
    # self.BATCH_SIZE = 128
    # self.EPOCHS = 150
    # self.OPT = 'sgd'
    # self.OPT_EXIST = True

  def conv(self, x_in, filters, kernel_size, strides=1, padding='same',
           use_bias=False, use_bn=True, activation='relu', kernel_initializer='he_normal'):
    '''卷积层'''
    self._COUNT_CONV += 1
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               kernel_initializer=kernel_initializer,
               name='CONV_' + str(self._COUNT_CONV))(x_in)
    if use_bn: x = BatchNormalization(name='BN_' + str(self._COUNT_CONV))(x)
    x = Activation(activation, name='RELU_' + str(self._COUNT_CONV))(x)
    return x

  def build_model(self):
    x_in = Input(shape=self.INPUT_SHAPE)

    # 卷积部分
    x = x_in
    for i in range(self.NUM_LAYERS[0]): x = self.conv(x, self.CONV, 3)
    x = MaxPool2D(2, padding='same')(x)
    for i in range(self.NUM_LAYERS[1]): x = self.conv(x, self.CONV, 3)
    x = MaxPool2D(2, padding='same')(x)
    for i in range(self.NUM_LAYERS[2]): x = self.conv(x, self.CONV, 3)
    x = MaxPool2D(2, padding='same')(x)
    for i in range(self.NUM_LAYERS[3]): x = self.conv(x, self.CONV, 3)
    x = MaxPool2D(self.INPUT_SHAPE[0] // 16 >= 4 and 2 or 1, padding='same')(x)
    for i in range(self.NUM_LAYERS[4]): x = self.conv(x, self.CONV, 3)
    if self.INPUT_SHAPE[0] // 32 >= 4:
      x = MaxPool2D(2, padding='same')(x)
    else:
      x = GlobalAvgPool2D()(x)
    
    # 全连接部分
    x = Flatten()(x)
    x = Dense(self.LOCAL, activation='relu')(x)
    x = Dropout(self.DROP)(x)
    x = Dense(self.NUM_CLASSES, activation='softmax')(x)

    self.model = Model(inputs=x_in, outputs=x, name='nvgg16')

# test part
if __name__ == "__main__":
  mod = nvgg16(DATAINFO={'INPUT_SHAPE': (224, 224, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
