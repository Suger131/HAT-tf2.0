"""
  MobileNet-v2-50
  NOTE:
    模型默认输入尺寸是(None, 300, 300, 3)
    模型中间特征图的设计参照Googlenet-v4
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           2,042,770
    Trainable params:       2,011,330
    Non-trainable params:   31,440
"""

# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *
from hat.models.advance import AdvNet


class mobilenetv2(AdvNet):
  """
    MobileNet-v2
  """
  def args(self):
    self.IRB_TIME = [1, 2, 3, 4, 3, 2, 1]
    self.IRB_CONV = [16, 24, 32, 64, 96, 160, 320]
    self.IRB_EXPD = [1, 6, 6, 6, 6, 6, 6]
    self.IRB_STEP = [1, 2, 2, 1, 2, 2, 1]
    self.IRB_CROP = [1, 1, 0, 0, 0, 0, 0]  # 1->True, 0->False
    self.IRB_CPDN = [0, 0, 1, 0, 1, 1, 0]  # 1->True, 0->False
    
    self.CONV = [32, 1280]
    
  def build_model(self):

    _list = list(zip(
      self.IRB_TIME,
      self.IRB_CONV,
      self.IRB_EXPD,
      self.IRB_STEP,
      self.IRB_CROP,
      self.IRB_CPDN
    ))

    x_in = self.input(self.INPUT_SHAPE)

    x = self.conv_bn(x_in, self.CONV[0], 3, 2)

    for i in _list:
      x = self._inverted_residual_block(x, *i)

    x = self.conv_bn(x, self.CONV[1], 1, 1)
    x = self.GAPool(x)
    x = self.reshape(x, (1, 1, self.CONV[1]))
    x = self.conv(x, self.NUM_CLASSES, 1, activation='softmax', name='Softmax')  # 加速
    x = self.flatten(x)

    return self.Model(inputs=x_in, outputs=x, name='mobilenetv2')

  def _bottleneck(self, x_in, filters, k, strides=1, first=False, crop=False, crop_down=False):

    if K.image_data_format() == 'channels_first':
      axis = 1
    else:
      axis = -1

    c = K.int_shape(x_in)[axis] * k

    x = self.conv(x_in, c, 1)
    x = self.bn(x)
    x = self.relu(x)
    x = DepthwiseConv2D(3, strides, padding='valid' if crop or crop_down else 'same')(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x, filters, 1)
    x = self.bn(x)

    if first:
      x_in = self.conv(x_in, filters, 1, strides)
      x_in = self.bn(x_in)
    if crop or crop_down:
      _crop = tuple([(int(i / 2), i - int(i / 2)) for i in [_in[j] - _nx[j] for j in (1, 2) for _in, _nx in ((K.int_shape(x_in), K.int_shape(x)),)]])
      x_in = Cropping2D(cropping=_crop)(x_in)
    x = self.add([x_in, x])
    x = self.relu(x)

    return x

  def _inverted_residual_block(self, x, n, filters, k, strides=1, crop=False, crop_down=False):

    x = self._bottleneck(x, filters, k, strides, first=True, crop_down=crop_down)

    x = self.repeat(self._bottleneck, n - 1, filters, k, crop=crop)(x)

    return x


# test part
if __name__ == "__main__":
  mod = mobilenetv2(DATAINFO={'INPUT_SHAPE': (300, 300, 3), 'NUM_CLASSES': 10}, built=True)
  mod.summary()
