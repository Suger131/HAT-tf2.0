"""
  StageNet
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           563,594
    Trainable params:       561,674
    Non-trainable params:   1,920
"""

# pylint: disable=no-name-in-module

from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import SGD
from hat.models.advance import AdvNet


class stagenet(AdvNet):
  """
    StageNet
  """
  def args(self):
    self.CONV_KI = 'he_normal'
    self.CONV_KR = l2(0.001)
    
    self.D = 0.25
    self.DX = 0.5

    self.EPOCHS = 384
    self.BATCH_SIZE = 128
    self.OPT = SGD(lr=1e-2, decay=1e-6)

  def build_model(self):

    x_in = self.input(self.INPUT_SHAPE)
    # (32,32, 3)
    x = self.conv(x_in, 16, 5, 2, kernel_initializer=self.CONV_KI, kernel_regularizer=self.CONV_KR)
    # (16,16,16)
    x = self.conv_bn(x, 32, 3, kernel_initializer=self.CONV_KI, kernel_regularizer=self.CONV_KR)
    x = self.conv_bn(x, 32, 3, 2, kernel_initializer=self.CONV_KI, kernel_regularizer=self.CONV_KR)
    x = self.dropout(x, self.D)
    # (8, 8, 32)
    x = self.conv_bn(x, 64, 3, kernel_initializer=self.CONV_KI, kernel_regularizer=self.CONV_KR)
    x = self.conv_bn(x, 64, 3, 2, kernel_initializer=self.CONV_KI, kernel_regularizer=self.CONV_KR)
    x = self.dropout(x, self.D)
    # (4, 4, 64)
    x = self.conv_bn(x, 128, 3, kernel_initializer=self.CONV_KI, kernel_regularizer=self.CONV_KR)
    x = self.conv_bn(x, 128, 3, 2, kernel_initializer=self.CONV_KI, kernel_regularizer=self.CONV_KR)
    x = self.dropout(x, self.D)
    # (2, 2, 128)
    x = self.flatten(x)
    x = self.local(x, 512, activation=None, kernel_initializer=self.CONV_KI)
    x = self.bn(x)
    x = self.relu(x)
    x = self.dropout(x, self.DX)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='stagenet')


# test part
if __name__ == "__main__":
  mod = stagenet(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10}, built=True)
  mod.summary()
