"""
  VGG-SE-16
  本模型默认总参数量[参考基准：car10]：
    Total params:           11,075,586
    Trainable params:       11,067,778
    Non-trainable params:   7,808
"""


from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras.models import Model


class vggse16(NetWork, AdvNet):
  """
    VGG-SE-16
  """

  def args(self):

    self.CONV = [64, 128, 192, 256, 384, 512]
    self.CONV_SIZE = 3

    self.POOL_SIZE = 3
    self.POOL_STRIDES = 2

    self.LOCAL = 1000
    self.DROP = 0.5

  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)
    
    # input (300,300,3)
    x = self.conv_se(x_in, self.CONV[0])
    x = self.conv_se(x, self.CONV[0])
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES)
    
    # (150,150,64)
    x = self.conv_se(x, self.CONV[1], padding='valid')
    # (148,148,128)
    x = self.conv_se(x, self.CONV[1])
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES)
    
    # (74,74,128)
    x = self.conv_se(x, self.CONV[2], padding='valid')
    # (72,72,256)
    x = self.conv_se(x, self.CONV[2])
    x = self.conv_se(x, self.CONV[2], padding='valid')
    # (70,70,256)
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES)
    
    # (35,35,256)
    x = self.repeat(self.conv_se, 3, x, self.CONV[3])
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES, padding='valid')
    
    # (17,17,384)
    x = self.repeat(self.conv_se, 3, x, self.CONV[4])
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES, padding='valid')
    
    # (8,8,512)
    x = self.repeat(self.conv_bn, 2, x, self.CONV[5], self.CONV_SIZE)
    
    # (8,8,1024)
    x = self.GAPool(x)
    x = self.local(x, self.LOCAL)
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='vggse16')

  def conv_se(self, x, filters, *args, **kwargs):
    x = self.conv_bn(x, filters, self.CONV_SIZE, *args, **kwargs)
    x = self.SE(x)
    return x


# test part
if __name__ == "__main__":
  mod = vggse16(DATAINFO={'INPUT_SHAPE': (300, 300, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
