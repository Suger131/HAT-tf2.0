'''
  DenseNet-39-BC
  B: BottleNeck
  C: Compression(Theta!=1)
  For Cifar10
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           175,358
    Trainable params:       170,466
    Non-trainable params:   4,892
'''


from tensorflow.python.keras import backend as K
from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras.models import Model


class densenet39(NetWork, AdvNet):
  """
    DenseNet-39-BC
  """
  def args(self):
    self.K = 12
    self.THETA = 0.5

    self.CONV = 20
    self.BLOCKS = [5, 6, 7]
  
  def build_model(self):
    x_in = self.input(self.IMAGE_SHAPE)

    x = self.conv(x_in, self.CONV, 3)  #1
    x = self.repeat(self._dense, self.BLOCKS[0], x, self.K)  #2*n 10 80->40
    x = self._transition(x)  #1
    x = self.repeat(self._dense, self.BLOCKS[1], x, self.K)  #2*n 12 40+72->56
    x = self._transition(x)  #1
    x = self.repeat(self._dense, self.BLOCKS[2], x, self.K)  #2*n 14 56+84=140
    
    x = self.bn(x)
    x = self.relu(x)
    x = self.GAPool(x)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='densenet39')

  def _dense(self, x_in, k):

    x = self.bn(x_in)
    x = self.relu(x)
    x = self.conv(x, 4 * k, 1, use_bias=False)
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x, k, 3, use_bias=False)

    x = self.concat([x_in, x])

    return x

  def _transition(self, x_in, strides=2):

    filters = int(K.int_shape(x_in)[-1] * self.THETA)

    x = self.bn(x_in)
    x = self.relu(x)
    x = self.conv(x, filters, 1, use_bias=False)

    x = self.avgpool(x, 2, strides)

    return x


# test part
if __name__ == "__main__":
  mod = densenet39(DATAINFO={'IMAGE_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.IMAGE_SHAPE)
  print(mod.model.summary())
