"""
  VGG-10
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           28,155,018
    Trainable params:       28,149,514
    Non-trainable params:   5,504
"""


from hat.models.advance import AdvNet
from tensorflow.python.keras.optimizers import SGD


class vgg10(AdvNet):
  """
    VGG-10
  """
  
  def args(self):
    self.TIME = [1, 1, 2, 2, 2]
    self.CONV = [64, 128, 256, 512, 512]
    self.LOCAL = [4096, 4096]
    self.DROP = 0.3
    
    self.OPT = SGD(lr=1e-3, momentum=.9)

  def build_model(self):

    x_in = self.input(self.INPUT_SHAPE)
    x = x_in

    # conv part
    for i in list(zip(self.TIME, self.CONV)):
      x = self.repeat(self.conv_bn, i[0], i[1], 3)(x)
      x = self.maxpool(x, 3, 2)

    # local part
    x = self.flatten(x)
    for i in self.LOCAL:
      x = self.local(x, i)
      x = self.dropout(x, self.DROP)
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='vgg10')

# test part
if __name__ == "__main__":
  mod = vgg10(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10}, built=True)
  mod.summary()
