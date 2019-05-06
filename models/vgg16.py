'''
  VGG16 模型
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           37,716,930
    Trainable params:       37,708,482
    Non-trainable params:   8,448
  本模型默认总参数量[参考基准：ImageNet]：
    Total params:           138,380,226
    Trainable params:       138,371,778
    Non-trainable params:   8,448
'''

from models.network import NetWork
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *


class vgg16(NetWork):
  """
  VGG16 模型
  """
  
  def args(self):
    self._COUNT_CONV = 0

    self.NUM_LAYERS = [2, 2, 3, 3, 3]
    self.CONV = [64, 128, 256, 512, 512]

    self.LOCAL = [4096, 4096, 1000]
    self.DROP = [0.3, 0.3, 0]
    self.LOCAL_LIST = list(zip(self.LOCAL, self.DROP))

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
    for i in range(self.NUM_LAYERS[0]): x = self.conv(x, self.CONV[0], 3)
    x = MaxPool2D(2, padding='same')(x)
    for i in range(self.NUM_LAYERS[1]): x = self.conv(x, self.CONV[1], 3)
    x = MaxPool2D(2, padding='same')(x)
    for i in range(self.NUM_LAYERS[2]): x = self.conv(x, self.CONV[2], 3)
    x = MaxPool2D(2, padding='same')(x)
    for i in range(self.NUM_LAYERS[3]): x = self.conv(x, self.CONV[3], 3)
    x = MaxPool2D(self.INPUT_SHAPE[0] // 16 >= 4 and 2 or 1, padding='same')(x)
    for i in range(self.NUM_LAYERS[4]): x = self.conv(x, self.CONV[4], 3)
    if self.INPUT_SHAPE[0] // 32 >= 4:
      x = MaxPool2D(2, padding='same')(x)
    else:
      x = GlobalAvgPool2D()(x)
    
    # 全连接部分
    x = Flatten()(x)
    for i in self.LOCAL_LIST:
      x = Dense(i[0], activation='relu')(x)
      x = Dropout(i[1])(x)
    x = Dense(self.NUM_CLASSES, activation='softmax')(x)

    self.model = Model(inputs=x_in, outputs=x, name='vgg16')

# test part
if __name__ == "__main__":
  mod = vgg16(DATAINFO={'INPUT_SHAPE': (224, 224, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
