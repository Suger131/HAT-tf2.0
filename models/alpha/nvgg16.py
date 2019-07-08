'''
  NVGG16 模型
  input size range: (26, 64)
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           16,324,938
    Trainable params:       16,312,394
    Non-trainable params:   12,544
'''


# pylint: disable=no-name-in-module

from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import SGD
from hat.models.advance import AdvNet


class nvgg16(AdvNet):
  """
  NVGG16 模型
  """
  
  def args(self):

    self.CONV_KI = 'he_normal'
    self.CONV_KR = l2(0.0005)

    self.TIME = [2, 2, 3, 3, 3]
    self.CONV = [64, 128, 256, 512, 512]

    self.LOCAL = [1024, 1024]
    self.D = 0.2
    self.DX = 0.5

    self.BATCH_SIZE = 128
    self.EPOCHS = 384
    self.OPT = SGD(lr=1e-2, decay=1e-6)

  def build_model(self):

    # params
    self.CONV_ = list(zip(
      self.TIME,
      self.CONV,
    ))

    x_in = self.input(shape=self.INPUT_SHAPE)

    x = x_in
    
    for ix, i in enumerate(self.CONV_):
      if ix:
        x = self.maxpool(x, 3, 1 if ix == 4 else 2)
        x = self.dropout(x, self.D)
      x = self.repeat(self._conv, *i, 3)(x)
    
    x = self.GAPool(x)
    x = self.dropout(x, self.D)
    x = self._local(x, self.LOCAL[0])
    x = self._local(x, self.LOCAL[1], d=self.DX)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='nvgg16')

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
    if d: x = self.dropout(x, d)
    return x


# test part
if __name__ == "__main__":
  mod = nvgg16(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10}, built=True)
  mod.summary()
