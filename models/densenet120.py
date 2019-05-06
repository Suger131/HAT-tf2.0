'''
  DenseNet-120-BC
  B: BottleNeck
  C: Compression(Theta!=1)
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           1,095,282
    Trainable params:       1,062,212
    Non-trainable params:   33,070
  本模型默认总参数量[参考基准：ImageNet]：
    Total params:           1,095,282
    Trainable params:       1,062,212
    Non-trainable params:   33,070
'''


from tensorflow.python.keras import backend as K
from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras.models import Model


class densenet120(NetWork, AdvNet):
  """
    DenseNet-120-BC
  """
  def args(self):
    self.K = 12
    self.THETA = 0.5

    self.CONV = 64
    self.CONV_SIZE = 7
    self.BLOCKS = [6, 12, 24, 16]
  
  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    x = self.conv(x_in, self.CONV, self.CONV_SIZE)  #1
    x = self.repeat(self._dense, self.BLOCKS[0], x, self.K)  #2*n 12
    x = self._transition(x)  #1
    x = self.repeat(self._dense, self.BLOCKS[1], x, self.K)  #2*n 24
    x = self._transition(x)  #1
    x = self.repeat(self._dense, self.BLOCKS[2], x, self.K)  #2*n 48
    x = self._transition(x)  #1
    x = self.repeat(self._dense, self.BLOCKS[3], x, self.K)  #2*n 32
    
    x = self.bn(x)
    x = self.relu(x)
    x = self.GAPool(x)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='densenet120')

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
  mod = densenet120(DATAINFO={'INPUT_SHAPE': (224, 224, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
