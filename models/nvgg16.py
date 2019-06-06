'''
  NVGG16 模型
  input size range: (26, 64)
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           37,757,922
    Trainable params:       37,731,090
    Non-trainable params:   26,832
'''


# pylint: disable=no-name-in-module
from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam


class nvgg16(NetWork, AdvNet):
  """
  NVGG16 模型
  """
  
  def args(self):

    self.CONV_KI = 'he_normal'
    self.CONV_KR = l2(0.0005)

    self.TIME = [2, 2, 3, 3, 3]
    self.CONV = [64, 128, 256, 512, 512]

    self.LOCAL = [4096, 4096, 1000]
    self.D = 0.25
    self.DX = 0.5

    # for test
    self.BATCH_SIZE = 128
    self.EPOCHS = 384
    self.OPT = Adam(lr=1e-3, decay=0.1 / 256)
    self.OPT_EXIST = True

  def build_model(self):

    x_in = self.input(shape=self.INPUT_SHAPE)

    x = x_in
    
    for i in range(len(self.CONV)):
      if i:
        x = self.maxpool(x, 3, 1 if i == 4 else 2)
        x = self.dropout(x, self.D)
      x = self.repeat(self._conv, self.TIME[i], x, self.CONV[i], 3)
    
    x = self.GAPool(x)
    x = self._local(x, self.LOCAL[0])
    x = self._local(x, self.LOCAL[1])
    x = self._local(x, self.LOCAL[2], d=self.DX)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='nvgg16')

  def _conv(self, x, *args, **kwargs):
    x = self.conv_bn(x, *args,
      kernel_initializer=self.CONV_KI,
      kernel_regularizer=self.CONV_KR, **kwargs)
    return x

  def _local(self, x, *args, d=None, **kwargs):
    if d == None: d = self.D
    x = self.local(x, *args,
      activation=None,
      kernel_initializer=self.CONV_KI, **kwargs)
    x = self.bn(x)
    x = self.relu(x)
    x = self.dropout(x, d)
    return x


# test part
if __name__ == "__main__":
  mod = nvgg16(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
