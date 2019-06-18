"""
  LeNet-3D
  For cifar10
  NOTE: INPUT_CHANNELS must > 1
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           1,800,222
    Trainable params:       1,800,222
    Non-trainable params:   0
"""

# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import

from tensorflow.python.keras.layers import *
from hat.models.advance import AdvNet


class lenet_3d(AdvNet):
  """
  LeNet-3D
  """
  
  def args(self):
    self.CONV = [20, 50]
    self.CONV3D_SIZE = (3, 3, 1)
    self.STRIDES = (3, 3, 1)
    self.LOCAL_SIZE = 500
    self.DROP_RATE = 0.5

    # self.BATCH_SIZE = 128
    # self.EPOCHS = 384
    self.OPT = 'sgd'

  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    # extand the rgb channel
    x = self.conv(x_in, 12, 5)
    # change 4D Tensor into 5D Tensor
    x = self.reshape(x, (self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], 12, 1))
    
    # 3D Conv
    x = self.conv3d(x, self.CONV[0], self.CONV3D_SIZE)
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv3d(x, self.CONV[0], self.CONV3D_SIZE)
    x = self.bn(x)
    x = self.relu(x)
    x = ZeroPadding3D((1, 1, 0))(x)
    x = MaxPool3D(self.STRIDES)(x)
    x = self.conv3d(x, self.CONV[1], self.CONV3D_SIZE)
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv3d(x, self.CONV[1], self.CONV3D_SIZE)
    x = self.bn(x)
    x = self.relu(x)
    x = ZeroPadding3D((1, 1, 0))(x)
    x = MaxPool3D(self.STRIDES)(x)
    
    # Local
    # x = AvgPool3D((8, 8, 1))(x)
    x = self.flatten(x)
    x = self.local(x, self.LOCAL_SIZE)
    x = self.dropout(x, self.DROP_RATE)
    x = self.local(x, self.LOCAL_SIZE)
    x = self.dropout(x, self.DROP_RATE)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.Model(inputs=x_in, outputs=x, name='lenet_3d')


# test part
if __name__ == "__main__":
  mod = lenet_3d(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
