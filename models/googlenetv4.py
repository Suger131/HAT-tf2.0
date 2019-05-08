"""
  GoogleNet-V4 模型
  本模型默认总参数量[参考基准：ImageNet]：
    Total params:           41,066,826
    Trainable params:       41,004,490
    Non-trainable params:   62,336
"""


from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *


class googlenetv4(NetWork, AdvNet):
  """
    GoogleNet-V4
  """
  def args(self):
    self.PAD = (38, 38)
  
  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    # Input (224,224,3) -> (300,300,3)
    # x = ZeroPadding2D(self.PAD)(x_in)
    # NOTE: default input shape is (300,300,3), not (224,224,3)
    x = x_in

    # Stem  (300,300,3) -> (35,35,384)
    x = self._Stem(x)

    # Inception-A (35,35,384)
    x = self.repeat(self._InceptionA, 4, x)

    # Reduction-A (35,35,384) -> (17,17,1024)
    x = self._ReductionA(x)

    # Inception-B (17,17,1024)
    x = self.repeat(self._InceptionB, 7, x)

    # Reduction-A (17,17,1024) -> (8,8,1536)
    x = self._ReductionB(x)

    # Inception-C (8,8,1536)
    x = self.repeat(self._InceptionC, 3, x)

    # Output
    x = self.GAPool(x)
    x = self.dropout(x, 0.2)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='googlenetv4')

  def _Stem(self, x_in):
    
    x = self.conv_bn(x_in, 32, 3, 2)
    x = self.conv_bn(x, 32, 3, padding='valid')
    x = self.conv_bn(x, 64, 3, padding='valid')
    
    x1 = self.maxpool(x, 3, 2)
    x2 = self.conv(x, 96, 3, 2, activation='relu')
    x = self.concat([x1, x2])
    
    x1 = self.conv_bn(x, 64, 1)
    x1 = self.conv_bn(x1, 64, (7, 1))
    x1 = self.conv_bn(x1, 64, (1, 7))
    x1 = self.conv_bn(x1, 96, 3, padding='valid')
    x2 = self.conv_bn(x, 64, 1)
    x2 = self.conv_bn(x2, 96, 3, padding='valid')
    x = self.concat([x1, x2])

    x1 = self.maxpool(x, 3, 2, padding='valid')
    x2 = self.conv(x, 192, 3, 2, padding='valid', activation='relu')
    x = self.concat([x1, x2])

    return x

  def _InceptionA(self, x_in):
    
    x1 = self.avgpool(x_in, 2, 1)
    x1 = self.conv_bn(x1, 96, 1)

    x2 = self.conv_bn(x_in, 96, 1)

    x3 = self.conv_bn(x_in, 64, 1)
    x3 = self.conv_bn(x3, 96, 3)

    x4 = self.conv_bn(x_in, 64, 1)
    x4 = self.conv_bn(x4, 64, 3)
    x4 = self.conv_bn(x4, 96, 3)
    
    x = self.concat([x1, x2, x3, x4])

    return x

  def _ReductionA(self, x_in):
    
    x1 = self.conv_bn(x_in, 192, 1)
    x1 = self.conv_bn(x1, 224, 3)
    x1 = self.conv_bn(x1, 256, 3, 2, padding='valid')

    x2 = self.conv_bn(x_in, 384, 3, 2, padding='valid')

    x3 = self.maxpool(x_in, 3, 2, padding='valid')

    x = self.concat([x1, x2, x3])
    
    return x

  def _InceptionB(self, x_in):
    
    x1 = self.avgpool(x_in, 2, 1)
    x1 = self.conv_bn(x1, 128, 1)

    x2 = self.conv_bn(x_in, 384, 1)

    x3 = self.conv_bn(x_in, 192, 1)
    x3 = self.conv_bn(x3, 224, (7, 1))
    x3 = self.conv_bn(x3, 256, (1, 7))

    x4 = self.conv_bn(x_in, 192, 1)
    x4 = self.conv_bn(x4, 192, (1, 7))
    x4 = self.conv_bn(x4, 224, (7, 1))
    x4 = self.conv_bn(x4, 224, (1, 7))
    x4 = self.conv_bn(x4, 256, (7, 1))
    
    x = self.concat([x1, x2, x3, x4])

    return x

  def _ReductionB(self, x_in):
    
    x1 = self.conv_bn(x_in, 256, 1)
    x1 = self.conv_bn(x1, 256, (1, 7))
    x1 = self.conv_bn(x1, 320, (7, 1))
    x1 = self.conv_bn(x1, 320, 3, 2, padding='valid')

    x2 = self.conv_bn(x_in, 192, 1)
    x2 = self.conv_bn(x2, 192, 3, 2, padding='valid')

    x3 = self.maxpool(x_in, 3, 2, padding='valid')

    x = self.concat([x1, x2, x3])
    
    return x

  def _InceptionC(self, x_in):
    
    x1 = self.avgpool(x_in, 2, 1)
    x1 = self.conv_bn(x1, 256, 1)

    x2 = self.conv_bn(x_in, 256, 1)

    xa = self.conv_bn(x_in, 384, 1)
    x3 = self.conv_bn(xa, 256, (1, 3))
    x4 = self.conv_bn(xa, 256, (3, 1))

    xb = self.conv_bn(x_in, 384, 1)
    xb = self.conv_bn(xb, 448, (1, 3))
    xb = self.conv_bn(xb, 512, (3, 1))
    x5 = self.conv_bn(xb, 256, (1, 3))
    x6 = self.conv_bn(xb, 256, (3, 1))
    
    x = self.concat([x1, x2, x3, x4, x5, x6])

    return x


# test part
if __name__ == "__main__":
  mod = googlenetv4(DATAINFO={'INPUT_SHAPE': (300, 300, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
  # mod.model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
  # mod.model.save('googlenetv4.h5', include_optimizer=False)
