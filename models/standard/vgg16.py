"""
  VGG-16
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           37,721,154
    Trainable params:       37,712,706
    Non-trainable params:   8,448
"""

# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import

from tensorflow.python.keras.models import Model
from hat.models.advance import AdvNet


class vgg16(AdvNet):
  """
    VGG-16
  """
  
  def args(self):
    self.TIME = [2, 2, 3, 3, 3]
    self.CONV = [64, 128, 256, 512, 512]
    self.POOL = [2, 2, 2, 2, 'g']
    self.LOCAL = [4096, 4096, 1000]
    self.DROP = [0.3, 0.3, 0]
    
    self.BATCH_SIZE = 128
    self.EPOCHS = 384
    self.OPT = 'adam'

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

    self.model = Model(inputs=x_in, outputs=x, name='vgg16')

# test part
if __name__ == "__main__":
  mod = vgg16(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
