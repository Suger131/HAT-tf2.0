'''
  WRN-28-10
  本模型默认总参数量[参考基准：cifar10]：
  S = 16
  K = 10
  N = 3
    Total params:           36,497,146
    Trainable params:       36,479,194
    Non-trainable params:   17,952
'''

# pylint: disable=no-name-in-module

from tensorflow.python.keras.optimizers import SGD
from hat.models.advance import AdvNet


class wrn28_10(AdvNet):
  """
  WRN-28-10 模型
  """

  def args(self):

    # main args
    self.S = 16
    self.K = 10
    self.N = 3
    self.D = 0

    # train args
    self.EPOCHS = 384
    self.BATCH_SIZE = 128
    self.OPT = SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)


  def build_model(self):

    # params
    Fs = [x * self.S * self.K for x in [1, 2, 4]]

    x_in = self.input(shape=self.INPUT_SHAPE)

    # Conv
    x = self.conv(x_in, self.S, 3)

    # Res Blocks
    x = self.conv_block(x, Fs[0], self.N, strides=1)
    x = self.conv_block(x, Fs[1], self.N)
    x = self.conv_block(x, Fs[2], self.N)

    # Output
    x = self.bn(x)
    x = self.relu(x)
    x = self.GAPool(x)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='wrn28_10')

  def expand(self, x_in, F, strides=2):
    """
      塑形层，要确保使用之前紧接着的是卷积层
    """
    xi = self.bn(x_in)
    xi = self.relu(xi)
    
    x_aux = self.conv(xi, F, 3, strides=strides)

    x = self.conv(xi, F, 3, strides=strides)
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x, F, 3)

    x = self.add([x, x_aux])

    return x

  def _block(self, x_in, F):
    """
      残差单元
    """
    x = self.bn(x_in)
    x = self.relu(x)
    x = self.conv(x, F, 3)

    x = self.dropout(x, self.D)

    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x, F, 3)

    x = self.add([x, x_in])
    
    return x

  def conv_block(self, x_in, F, N, strides=2):
    """
      残差块
    """
    x = self.expand(x_in, F, strides=strides)

    x = self.repeat(self._block, N, F)(x)

    return x


# test part
if __name__ == "__main__":
  mod = wrn28_10(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10}, built=True)
  mod.summary()
