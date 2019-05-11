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


from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *


class mobilenetv2(NetWork, AdvNet):
  """
    MobileNet-v2
  """
  def args(self):
    pass

  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    x = x_in
    x = self.conv_bn(x, 32, 3, 2)
    x = self._inverted_residual_block(x, 1, 16, 1, strides=1, crop=True)
    x = self._inverted_residual_block(x, 2, 24, 6, strides=2, crop=True)
    x = self._inverted_residual_block(x, 3, 32, 6, strides=2, crop_down=True)
    x = self._inverted_residual_block(x, 4, 64, 6, strides=1)
    x = self._inverted_residual_block(x, 3, 96, 6, strides=2, crop_down=True)
    x = self._inverted_residual_block(x, 2, 160, 6, strides=2, crop_down=True)
    x = self._inverted_residual_block(x, 1, 320, 6, strides=1)
    x = self.conv_bn(x, 1280, 1, 1)
    x = self.GAPool(x)
    x = self.reshape(x, (1, 1, 1280))
    x = self.conv(x, self.NUM_CLASSES, 1, activation='softmax', name='Softmax')
    x = self.flatten(x)

    self.model = Model(inputs=x_in, outputs=x, name='vggse16')

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
      _in = K.int_shape(x_in)
      _nx = K.int_shape(x)
      h = _in[1] - _nx[1]
      w = _in[2] - _nx[2]
      _crop = ((int(h/2), h - int(h/2)), (int(w/2), w - int(w/2)))
      x_in = Cropping2D(cropping=_crop)(x_in)
    # print(K.int_shape(x_in), K.int_shape(x))
    x = self.add([x_in, x])
    x = self.relu(x)

    return x

  def _inverted_residual_block(self, x, n, filters, k, strides=1, crop=False, crop_down=False):

    x = self._bottleneck(x, filters, k, strides, first=True, crop=crop, crop_down=crop_down)

    for i in range(n - 1):
      x = self._bottleneck(x, filters, k, crop=crop)

    return x


# test part
if __name__ == "__main__":
  mod = mobilenetv2(DATAINFO={'INPUT_SHAPE': (300, 300, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
