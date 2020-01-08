"""
  CNN64 模型
  本模型默认总参数量[参考基准：fruits]：
    Total params:           1,632,080
    Trainable params:       1,632,080
    Non-trainable params:   0
"""


from hat.models.advance import AdvNet
from tensorflow.python.keras.optimizers import SGD


class cnn64(AdvNet):
  """
    CNN64
  """

  def args(self):

    self.CONV = [16, 32, 64, 128]
    self.LOCAL = [1024, 256]
    self.DROP = 0.3

    self.OPT = SGD(lr=1e-3, momentum=.9)
    
  def build_model(self):

    x_in = self.input(self.INPUT_SHAPE)

    # conv
    x = x_in
    for i in self.CONV:
      x = self.conv(x, i, 5, activation='relu')
      x = self.maxpool(x)

    # local
    x = self.flatten(x)
    for i in self.LOCAL:
      x = self.local(x, i)
      x = self.dropout(x, self.DROP)

    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='cnn64')


# test part
if __name__ == "__main__":
  mod = cnn64(DATAINFO={'INPUT_SHAPE': (100, 100, 3), 'NUM_CLASSES': 114}, built=True)
  mod.summary()
