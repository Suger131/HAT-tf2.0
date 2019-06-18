'''
  DenseNet-BC
  B: BottleNeck
  C: Compression(Theta!=1)

  本模型默认总参数量[参考基准：cifar10]：
    Total params:           1,095,282
    Trainable params:       1,062,212
    Non-trainable params:   33,070
'''

# pylint: disable=no-name-in-module

from tensorflow.python.keras import backend as K
from hat.models.advance import AdvNet


# import setting
__all__ = [
  'densenet',
  'densenet121',
  'densenet169',
  'densenet201',
  'densenet264',
  'densenet121_24',
  'densenet169_24',
  'densenet201_24',
  'densenet264_24',
  'densenet121_32',
  'densenet169_32',
  'densenet201_32',
  'densenet264_32',
]


class densenet(AdvNet):
  """
    DenseNet-BC
  """

  def __init__(self, blocks, k=12, theta=0.5, use_bias=False, name='', **kwargs):
    self.BLOCKS = blocks
    self.K = k
    self.THETA = theta
    self.BIAS = use_bias
    self.NAME = name
    super().__init__(**kwargs)

  def args(self):

    self.BLOCKS = self.BLOCKS
    self.K = self.K
    self.THETA = self.THETA
    self.BIAS = self.BIAS

    self.CONV = 64
    self.CONV_SIZE = 7
    self.CONV_STEP = 2 if self.INPUT_SHAPE[0] >= 64 else 1
  
  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    x = self.conv(x_in, self.CONV, self.CONV_SIZE, self.CONV_STEP)  #1
    
    x = self.repeat(self._dense, self.BLOCKS[0], self.K)(x)  #2*n 12
    x = self._transition(x)  #1
    x = self.repeat(self._dense, self.BLOCKS[1], self.K)(x)  #2*n 24
    x = self._transition(x)  #1
    x = self.repeat(self._dense, self.BLOCKS[2], self.K)(x)  #2*n 48
    x = self._transition(x)  #1
    x = self.repeat(self._dense, self.BLOCKS[3], self.K)(x)  #2*n 32
    
    x = self.bn(x)
    x = self.relu(x)
    x = self.GAPool(x)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.Model(inputs=x_in, outputs=x, name=self.NAME)

  def _dense(self, x_in, k):

    x = self.bn(x_in)
    x = self.relu(x)
    x = self.conv(x, 4 * k, 1, use_bias=self.BIAS)
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x, k, 3, use_bias=self.BIAS)

    x = self.concat([x_in, x])

    return x

  def _transition(self, x_in, strides=2):

    filters = int(K.int_shape(x_in)[-1] * self.THETA)

    x = self.bn(x_in)
    x = self.relu(x)
    x = self.conv(x, filters, 1, use_bias=self.BIAS)

    x = self.avgpool(x, 2, strides)

    return x


# Models
def densenet121(**kwargs):
  """
    DenseNet-121-BC
    ```
    Blocks: [6, 12, 24, 16]
    k     : 12
    theta : 0.5
    bias  : False
    ```
  """
  return densenet(
    blocks=[6, 12, 24, 16],
    k=12,
    theta=0.5,
    use_bias=False,
    name='densenet121',
    **kwargs
  )


def densenet169(**kwargs):
  """
    DenseNet-169-BC
    ```
    Blocks: [6, 12, 32, 32]
    k     : 12
    theta : 0.5
    bias  : False
    ```
  """
  return densenet(
    blocks=[6, 12, 32, 32],
    k=12,
    theta=0.5,
    use_bias=False,
    name='densenet169',
    **kwargs
  )


def densenet201(**kwargs):
  """
    DenseNet-169-BC
    ```
    Blocks: [6, 12, 48, 32]
    k     : 12
    theta : 0.5
    bias  : False
    ```
  """
  return densenet(
    blocks=[6, 12, 48, 32],
    k=12,
    theta=0.5,
    use_bias=False,
    name='densenet201',
    **kwargs
  )


def densenet264(**kwargs):
  """
    DenseNet-264-BC
    ```
    Blocks: [6, 12, 64, 48]
    k     : 12
    theta : 0.5
    bias  : False
    ```
  """
  return densenet(
    blocks=[6, 12, 64, 48],
    k=12,
    theta=0.5,
    use_bias=False,
    name='densenet264',
    **kwargs
  )


def densenet121_24(**kwargs):
  """
    DenseNet-121-BC
    ```
    Blocks: [6, 12, 24, 16]
    k     : 24
    theta : 0.5
    bias  : False
    ```
  """
  return densenet(
    blocks=[6, 12, 24, 16],
    k=24,
    theta=0.5,
    use_bias=False,
    name='densenet121_24',
    **kwargs
  )


def densenet169_24(**kwargs):
  """
    DenseNet-169-BC
    ```
    Blocks: [6, 12, 32, 32]
    k     : 24
    theta : 0.5
    bias  : False
    ```
  """
  return densenet(
    blocks=[6, 12, 32, 32],
    k=24,
    theta=0.5,
    use_bias=False,
    name='densenet169_24',
    **kwargs
  )


def densenet201_24(**kwargs):
  """
    DenseNet-169-BC
    ```
    Blocks: [6, 12, 48, 32]
    k     : 24
    theta : 0.5
    bias  : False
    ```
  """
  return densenet(
    blocks=[6, 12, 48, 32],
    k=24,
    theta=0.5,
    use_bias=False,
    name='densenet201_24',
    **kwargs
  )


def densenet264_24(**kwargs):
  """
    DenseNet-264-BC
    ```
    Blocks: [6, 12, 64, 48]
    k     : 24
    theta : 0.5
    bias  : False
    ```
  """
  return densenet(
    blocks=[6, 12, 64, 48],
    k=24,
    theta=0.5,
    use_bias=False,
    name='densenet264_24',
    **kwargs
  )


def densenet121_32(**kwargs):
  """
    DenseNet-121-BC
    ```
    Blocks: [6, 12, 24, 16]
    k     : 32
    theta : 0.5
    bias  : False
    ```
  """
  return densenet(
    blocks=[6, 12, 24, 16],
    k=32,
    theta=0.5,
    use_bias=False,
    name='densenet121_32',
    **kwargs
  )


def densenet169_32(**kwargs):
  """
    DenseNet-169-BC
    ```
    Blocks: [6, 12, 32, 32]
    k     : 32
    theta : 0.5
    bias  : False
    ```
  """
  return densenet(
    blocks=[6, 12, 32, 32],
    k=32,
    theta=0.5,
    use_bias=False,
    name='densenet169_32',
    **kwargs
  )


def densenet201_32(**kwargs):
  """
    DenseNet-169-BC
    ```
    Blocks: [6, 12, 48, 32]
    k     : 32
    theta : 0.5
    bias  : False
    ```
  """
  return densenet(
    blocks=[6, 12, 48, 32],
    k=32,
    theta=0.5,
    use_bias=False,
    name='densenet201_32',
    **kwargs
  )


def densenet264_32(**kwargs):
  """
    DenseNet-264-BC
    ```
    Blocks: [6, 12, 64, 48]
    k     : 32
    theta : 0.5
    bias  : False
    ```
  """
  return densenet(
    blocks=[6, 12, 64, 48],
    k=32,
    theta=0.5,
    use_bias=False,
    name='densenet264_32',
    **kwargs
  )


# test part
if __name__ == "__main__":
  mod = densenet121_32(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
