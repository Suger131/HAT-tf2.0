"""
  VGG-Swish-SE
  For Cifar10
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           16,402,790
    Trainable params:       16,390,246
    Non-trainable params:   12,544
"""


from hat.models.advance import AdvNet


class vss(AdvNet):
  """
    VSS
  """
  def args(self):
    self.CONV_KI = 'he_normal'
    self.CONV_KR = None # l2(0.0005)

    self.TIME = [2, 2, 3, 3, 3]
    self.CONV = [64, 128, 256, 512, 512]

    self.BATCH_SIZE = 128
    self.EPOCHS = 384
    self.OPT = 'Adam'

  def build_model(self):

    # params processing
    self.CONV_ = list(zip(
      self.TIME,
      self.CONV,
    ))

    x_in = self.input(self.INPUT_SHAPE)

    x = x_in

    for ix, i in enumerate(self.CONV_):
      if ix: x = self.poolx(x)
      x = self.repeat(self.conv_s, *i)(x)
      x = self.SE(x)

    x = self.GAPool(x)

    x = self.local_s(x, 1024)
    x = self.dropout(x, 0.3 )
    x = self.local_s(x, 1024)
    x = self.dropout(x, 0.3 )

    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='vss')

  def conv_s(self, x_in, filters, kernel_size=3):
    
    x = self.conv(
      x_in, 
      filters, 
      kernel_size, 
      kernel_initializer=self.CONV_KI,
      kernel_regularizer=self.CONV_KR
    )
    x = self.bn(x)
    x = self.swish(x)
    # x = self.SE(x)
    
    return x

  def local_s(self, x_in, units):

    x = self.local(
      x_in, 
      units, 
      activation=None,
      kernel_initializer=self.CONV_KI,
    )
    x = self.bn(x)
    x = self.swish(x)

    return x

  def poolx(self, x_in, pool_size=3, strides=2):

    maxx = self.maxpool(x_in, pool_size=pool_size, strides=strides)
    avgx = self.avgpool(x_in, pool_size=pool_size, strides=strides)
    x = self.add([maxx, avgx])

    return x


# test part
if __name__ == "__main__":
  mod = vss(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10}, built=True)
  mod.summary()
