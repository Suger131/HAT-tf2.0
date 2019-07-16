"""
  Shuffle Net
  本模型默认总参数量[参考基准：car10]：
    Total params:           160,786
    Trainable params:       155,666
    Non-trainable params:   5,120
"""

# pylint: disable=no-name-in-module

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda
from hat.models.advance import AdvNet
from hat.utils import Counter


class shufflenet(AdvNet):
  """
    Shuffle Net
  """

  def args(self):
    self.HEAD_CONV = [32, 64]
    
    self.STAGE_CONV = [64, 128, 256, 768]
    self.STAGE_TIME = [3, 7, 3]
    self.DROP = 0.5

  def build_model(self):
    
    self.axis = -1
    
    x_in = self.input(self.INPUT_SHAPE)
    
    x = self._head(x_in)

    x = self.stage(x, self.STAGE_CONV[0], self.STAGE_TIME[0])
    x = self.stage(x, self.STAGE_CONV[1], self.STAGE_TIME[1])
    x = self.stage(x, self.STAGE_CONV[2], self.STAGE_TIME[2])

    x = self.conv_bn(x, self.STAGE_CONV[3], 1)
    x = self.GAPool(x)
    x = self.dropout(x, self.DROP)

    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='shufflenet')

  def _head(self, x_in):
    
    x = self.conv_bn(x_in, 24, 3, 2)
    x = self.conv_bn(x, 24, 3)
    x = self.conv_bn(x, 48, 3, 2)
    x = self.conv_bn(x, 48, 3)
    return x

  def stage(self, x_in, filters, n, crop=True):
    
    n = n - 1
    x = self._block_d(x_in, filters, crop)
    x = self.repeat(self._block_c, n)(x)
    return x

  def _block_d(self, x_in, filters, crop=True):
    
    _channels = K.int_shape(x_in)[self.axis]
    _pad = 'valid' if crop else 'same'
    _hc = filters // 2

    x1 = self.dwconv(x_in, 3, 2)
    x1 = self.bn(x1)
    x1 = self.conv(x1, _hc, 1)

    x2 = self.conv(x_in, _hc, 1)
    x2 = self.bn(x2)
    x2 = self.relu(x2)
    x2 = self.dwconv(x2, 3, 2)
    x2 = self.bn(x2)
    x2 = self.conv(x2, _hc, 1)

    x = self.shuffle(self.axis)([x1, x2])
    x = self.bn(x)
    x = self.relu(x)
    
    return x

  def _block_c(self, x_in):
    
    _channels = K.int_shape(x_in)[self.axis]
    _hc = _channels // 2
    _side, x = self._split(x_in)

    x = self.conv(x, _hc, 1)
    x = self.bn(x)
    x = self.relu(x)
    x = self.dwconv(x, 3)
    x = self.bn(x)
    x = self.conv(x, _hc, 1)
    x = self.bn(x)
    x = self.relu(x)
    
    x = self.shuffle(self.axis)([_side, x])

    return x

  def _split(self, x_in):
    
    count = Counter('split')
    channels = K.int_shape(x_in)[self.axis]
    hc = channels // 2
    side = Lambda(lambda z:z[:,:,:,0:hc], name=f"Split_Side_{count}")(x_in)
    stem = Lambda(lambda z:z[:,:,:,hc: ], name=f"Split_Stem_{count}")(x_in)
    
    return side, stem


# test part
if __name__ == "__main__":
  mod = shufflenet(DATAINFO={'INPUT_SHAPE': (100, 100, 3), 'NUM_CLASSES': 114}, built=True)
  mod.summary()
