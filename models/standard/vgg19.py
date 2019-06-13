"""
  VGG-19
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           43,035,970
    Trainable params:       43,024,962
    Non-trainable params:   11,008
"""

# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import

from tensorflow.python.keras.models import Model
from hat.models.network import NetWork
from hat.models.advance import AdvNet


class vgg19(NetWork, AdvNet):
  """
    VGG-19
  """
  
  def args(self):
    self.TIME = [2, 2, 4, 4, 4]
    self.CONV = [64, 128, 256, 512, 512]
    self.POOL = [2, 2, 2, 2, 'g']
    self.LOCAL = [4096, 4096, 1000]
    self.DROP = [0.3, 0.3, 0]
    
    self.BATCH_SIZE = 128
    self.EPOCHS = 384
    self.OPT = 'adam'
    self.OPT_EXIST = True

  def build_model(self):
    
    # params processing
    self.CONV_ = list(zip(
      self.TIME,
      self.CONV,
      self.POOL
    ))
    self.LOCAL_ = list(zip(
      self.LOCAL,
      self.DROP))

    x_in = self.input(self.INPUT_SHAPE)
    x = x_in

    # conv part
    for i in self.CONV_:
      x = self.repeat(self.conv_bn, i[0], i[1], 3)(x)
      x = self.maxpool(x, 3, i[2]) if i[2] != 'g' else self.GAPool(x)

    # local part
    for i in self.LOCAL_:
      x = self.local(x, i[0])
      x = self.dropout(x, i[1]) if i[1] else x
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='vgg19')

# test part
if __name__ == "__main__":
  mod = vgg19(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
