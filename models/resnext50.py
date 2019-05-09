'''
  ResNeXt-50
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           25,646,722
    Trainable params:       25,593,602
    Non-trainable params:   53,120
  本模型默认总参数量[参考基准：ImageNet]：
    Total params:           25,646,722
    Trainable params:       25,593,602
    Non-trainable params:   53,120
'''


from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras.models import Model


class resnext50(NetWork, AdvNet):
  """
    ResNeXt-50
  """

  def args(self):
    self.CONV_F = 64
    self.CONV_SIZE = 7
    self.CONV_STRIDES = 2 if self.INPUT_SHAPE[0] // 16 >= 4 else 1
    self.POOL_SIZE = 3
    self.POOL_STRIDES = 2

    self.RES_TIMES = [3, 4, 6, 3]
    self.GROUP = 32
    self.GROUP_FILTERS = [4, 8, 16, 32]
    self.RES_FILTERS = [256, 512, 1024, 2048]
    self.RES_STRIDES = [1 if self.INPUT_SHAPE[0] // 16 >= 4 else 2,
                        2,
                        2,
                        2 if self.INPUT_SHAPE[0] // 32 >= 4 else 1]

    self.LOCAL = 1000

    # for test
    # self.BATCH_SIZE = 128
    # self.EPOCHS = 150
    # self.OPT = 'sgd'
    # self.OPT_EXIST = True

  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    # first conv
    x = self.conv_bn(x_in, self.CONV_F, self.CONV_SIZE, strides=self.CONV_STRIDES)
    if self.INPUT_SHAPE[0] // 16 >= 4: x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES)

    # res part
    _res_list = list(zip(self.RES_TIMES,
                         self.GROUP_FILTERS,
                         self.RES_FILTERS,
                         self.RES_STRIDES))
    for i in _res_list:
      x = self._block(x, i[0], self.GROUP, i[1], i[2], i[3])

    # local part
    x = self.GAPool(x)
    x = self.local(x, self.LOCAL)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='resnext50')

  def _block(self, x_in, times, groups, filters, channels, strides=2):
    x = x_in
    x = self._bottle(x, groups, filters, channels, strides=strides, _t=True)
    times -= 1
    x = self.repeat(self._bottle, times, x, groups, filters, channels)
    return x

  def _bottle(self, x_in, groups, filters, channels, strides=1, _t=False):

    x = self.conv(x_in, groups * filters, 1, strides=strides)
    x = self.bn(x)
    x = self.relu(x)
    x = self.groupconv(x, groups, filters, 3)
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x, channels, 1)
    x = self.bn(x)

    if _t:
      x_ = self.conv(x_in, channels, 1, strides=strides)
      x_ = self.bn(x_)
    else:
      x_ = x_in

    x = self.add([x, x_])
    x = self.relu(x)

    return x

# test part
if __name__ == "__main__":
  mod = resnext50(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
  mod.model.compile('adam',
      'sparse_categorical_crossentropy',
      metrics=['accuracy'])
  mod.model.save('resnext.h5', include_optimizer=False)
  