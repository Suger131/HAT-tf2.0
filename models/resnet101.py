'''
  ResNet-101
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           44,717,186
    Trainable params:       44,611,842
    Non-trainable params:   105,344
  本模型默认总参数量[参考基准：ImageNet]：
    Total params:           44,717,186
    Trainable params:       44,611,842
    Non-trainable params:   105,344
'''


from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras.models import Model


class resnet101(NetWork, AdvNet):
  """
    ResNet-101
  """

  def args(self):
    self.CONV_F = 64
    self.CONV_SIZE = 7
    self.CONV_STRIDES = 2
    self.POOL_SIZE = 3
    self.POOL_STRIDES = 2

    self.RES_TIMES = [3, 4, 23, 3]
    self.RES_F_A = [64, 128, 256, 512]
    self.RES_F_B = [256, 512, 1024, 2048]
    self.RES_STRIDES = [1, 2, 2, 2 if self.IMAGE_SHAPE[0] // 32 >= 4 else 1]

    self.LOCAL = 1000

    # for test
    # self.BATCH_SIZE = 128
    # self.EPOCHS = 150
    # self.OPT = 'sgd'
    # self.OPT_EXIST = True

  def build_model(self):
    x_in = self.input(self.IMAGE_SHAPE)

    # first conv
    x = self.conv_bn(x_in, self.CONV_F, self.CONV_SIZE, strides=self.CONV_STRIDES)
    if self.IMAGE_SHAPE[0] // 16 >= 4: x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES)

    # res part
    _res_list = list(zip(self.RES_TIMES,
                         self.RES_F_A,
                         self.RES_F_B,
                         self.RES_STRIDES))
    for i in _res_list:
      x = self._block(x, i[0], i[1], i[2], i[3])

    # local part
    x = self.GAPool(x)
    x = self.local(x, self.LOCAL)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='resnet101')

  def _block(self, x_in, times, filters1, filters2, strides=2):
    x = x_in
    x = self._bottle(x, filters1, filters2, strides=strides, _t=True)
    times -= 1
    x = self.repeat(self._bottle, times, x, filters1, filters2)
    return x

  def _bottle(self, x_in, filters1, filters2, strides=1, _t=False):

    x = self.conv(x_in, filters1, 1, strides=strides)
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x, filters1, 3)
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x, filters2, 1)
    x = self.bn(x)

    if _t:
      x_ = self.conv(x_in, filters2, 1, strides=strides)
      x_ = self.bn(x_)
    else:
      x_ = x_in

    x = self.add([x, x_])
    x = self.relu(x)

    return x

# test part
if __name__ == "__main__":
  mod = resnet101(DATAINFO={'IMAGE_SHAPE': (224, 224, 3), 'NUM_CLASSES': 10})
  print(mod.IMAGE_SHAPE)
  print(mod.model.summary())
