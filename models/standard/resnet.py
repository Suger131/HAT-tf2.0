"""
  ResNet 系列
  
  本模型默认总参数量[参考基准：cifar10]：

    【resnet50】
    Total params:           25,631,362
    Trainable params:       25,585,922
    Non-trainable params:   45,440
    -------------------------------------------------
    【resnet101】
    Total params:           44,701,826
    Trainable params:       44,604,162
    Non-trainable params:   97,664
    ------------------------------------------------- 
    【resnet152】
    Total params:           60,414,594
    Trainable params:       60,270,850
    Non-trainable params:   143,744
    -------------------------------------------------
   【resnetse50】
    Total params:           28,162,354
    Trainable params:       28,116,914
    Non-trainable params:   45,440
    -------------------------------------------------
   【resnetse101】
    Total params:           49,479,538
    Trainable params:       49,381,874
    Non-trainable params:   97,664
    -------------------------------------------------
   【resnetse152】
    Total params:           67,043,634
    Trainable params:       66,899,890
    Non-trainable params:   143,744
    -------------------------------------------------
   【resnext50】
    Total params:           25,125,890
    Trainable params:       25,065,346
    Non-trainable params:   60,544
    -------------------------------------------------
   【resnext101】
    Total params:           44,379,138
    Trainable params:       44,248,962
    Non-trainable params:   130,176
    -------------------------------------------------
   【resnext152】
    Total params:           60,244,994
    Trainable params:       60,053,378
    Non-trainable params:   191,616
"""


from hat.models.advance import AdvNet


# import setting
__all__ = [
  'resnet',
  'resnet50',
  'resnet101',
  'resnet152',
  'resnetse50',
  'resnetse101',
  'resnetse152',
  'resnext50',
  'resnext101',
  'resnext152',
]


class resnet(AdvNet):
  """
    ResNet
  """

  def __init__(self, times, use_se=False, use_group=False, name='', **kwargs):
    self.RES_TIMES = times
    self.USE_SE = use_se
    self.USE_GROUP = use_group
    self.NAME = name
    super().__init__(**kwargs)

  def args(self):

    self.RES_TIMES = self.RES_TIMES
    self.USE_SE = self.USE_SE
    self.USE_GROUP = self.USE_GROUP

    self.CONV_F = 64
    self.CONV_SIZE = 7
    self.CONV_STRIDES = 2 if self.INPUT_SHAPE[0] // 16 >= 4 else 1
    self.POOL_SIZE = 3
    self.POOL_STRIDES = 2

    self.RES_FA = [64, 128, 256, 512]
    self.RES_FB = [256, 512, 1024, 2048]
    self.RES_STRIDES = [1, 2, 2, 2 if self.INPUT_SHAPE[0] // 32 >= 4 else 1]

    self.LOCAL = 1000
    self.DROP = 0.5

    # self.BATCH_SIZE = 128
    # self.EPOCHS = 384
    # self.OPT = 'sgd'

  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    # first conv
    x = self.conv(x_in, self.CONV_F, self.CONV_SIZE, strides=self.CONV_STRIDES)
    x = self.maxpool(x, self.POOL_SIZE, self.POOL_STRIDES)

    # res part
    _res_list = list(zip(self.RES_TIMES,
                         self.RES_FA,
                         self.RES_FB,
                         self.RES_STRIDES))
    for i in _res_list:
      x = self._block(x, *i)

    # local part
    x = self.bn(x)
    x = self.relu(x)
    x = self.GAPool(x)
    x = self.local(x, self.LOCAL)
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name=self.NAME)

  def _block(self, x_in, times, filters1, filters2, strides=2):
    x = self._bottle(x_in, filters1, filters2, strides=strides, _t=True)
    times -= 1
    x = self.repeat(self._bottle, times, filters1, filters2)(x)
    return x

  def _bottle(self, x_in, filters1, filters2, strides=1, _t=False):

    x = self.bn(x_in)
    x = self.relu(x)
    
    if _t:
      x_ = self.conv(x, filters2, 1)
      if strides != 1:
        x_ = self.maxpool(x_,  3, strides=strides)
    else:
      x_ = x_in

    if self.USE_GROUP:
      x = self.conv(x, filters1 * 2, 1)
      x = self.bn(x)
      x = self.relu(x)
      x = self.groupconv(x, 32, filters1 // 16, 3, strides=strides)
    else:
      x = self.conv(x, filters1, 1)
      x = self.bn(x)
      x = self.relu(x)
      x = self.conv(x, filters1, 3, strides=strides)
    
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x, filters2, 1)
    if self.USE_SE:
      x = self.SE(x)

    x = self.add([x, x_])
    
    return x


# Models
def resnet50(**kwargs):
  """
    ResNet-50
    
    Times: [3, 4, 6, 3]
  """
  return resnet(
    times=[3, 4, 6, 3],
    name='resnet50',
    **kwargs
  )


def resnet101(**kwargs):
  """
    ResNet-101
    
    Times: [3, 4, 23, 3]
  """
  return resnet(
    times=[3, 4, 23, 3],
    name='resnet101',
    **kwargs
  )


def resnet152(**kwargs):
  """
    ResNet-152
    
    Times: [3, 8, 36, 3]
  """
  return resnet(
    times=[3, 8, 36, 3],
    name='resnet152',
    **kwargs
  )


def resnetse50(**kwargs):
  """
    ResNet-SE-50
    
    Times: [3, 4, 6, 3]
  """
  return resnet(
    times=[3, 4, 6, 3],
    use_se=True,
    name='resnetse50',
    **kwargs
  )


def resnetse101(**kwargs):
  """
    ResNet-SE-101
    
    Times: [3, 4, 23, 3]
  """
  return resnet(
    times=[3, 4, 23, 3],
    use_se=True,
    name='resnetse101',
    **kwargs
  )


def resnetse152(**kwargs):
  """
    ResNet-SE-152
    
    Times: [3, 8, 36, 3]
  """
  return resnet(
    times=[3, 8, 36, 3],
    use_se=True,
    name='resnetse152',
    **kwargs
  )


def resnext50(**kwargs):
  """
    ResNeXt-50
    
    Times: [3, 4, 6, 3]
  """
  return resnet(
    times=[3, 4, 6, 3],
    use_group=True,
    name='resnext50',
    **kwargs
  )


def resnext101(**kwargs):
  """
    ResNeXt-101
    
    Times: [3, 4, 23, 3]
  """
  return resnet(
    times=[3, 4, 23, 3],
    use_group=True,
    name='resnext101',
    **kwargs
  )


def resnext152(**kwargs):
  """
    ResNeXt-152
    
    Times: [3, 8, 36, 3]
  """
  return resnet(
    times=[3, 8, 36, 3],
    use_group=True,
    name='resnext152',
    **kwargs
  )


# test part
if __name__ == "__main__":
  mod = resnext50(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10}, built=True)
  mod.summary()
