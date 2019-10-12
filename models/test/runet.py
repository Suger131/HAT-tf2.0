"""
  RU-Net
  Reuse Net
  本模型默认总参数量[参考基准：ImageNet]：
    Total params:           510,154
    Trainable params:       506,986
    Non-trainable params:   3,168
"""

# pylint: disable=no-name-in-module

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.optimizers import SGD, Adam
from hat.models.advance import AdvNet

class runet(AdvNet):
  """
    RU-Net
  """
  def args(self):
    self.OPT = Adam(lr=1e-4)
    # self.OPT = SGD(lr=1e-4, momentum=.9)#, decay=4e-4

  def build_model(self):

    x_in = self.input(self.INPUT_SHAPE)

    x = x_in

    x = self.conv_bn(x, 64, 5)
    x = self.conv_bn(x, 128, 5, 2)
    x = self.conv_bn(x, 256, 5, 2)
    x = self.RuStage(x, 2, 2, 256, 3)
    x = self.conv_bn(x, 512, 5, 2)
    x = self.RuStage(x, 2, 2, 512, 3)
    
    x = self.avgpool(x)
    x = self.flatten(x)
    x = self.local(x, 1024, activation='relu')
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    return self.Model(inputs=x_in, outputs=x, name='runet')

  def RuStage(self, x_in, n, r, filters, kernel_size):
    """RuStage"""

    x = x_in
    channels = K.int_shape(x_in)[-1]

    clist = []
    for i in range(n):
      clist.append(Conv2D(filters, kernel_size, padding='same', activation='relu'))

    if channels != filters:
      x = self.conv_bn(x, filters, kernel_size)

    for i in range(r):
      for j in range(n):
        x = clist[j](x)

    return x


# test
if __name__ == "__main__":
  mod = runet(DATAINFO={
    'INPUT_SHAPE': (32, 32, 3),
    'NUM_CLASSES': 10}, built=True)
  mod.summary()
  mod.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  )
  mod.save('runet.h5')
  # mod.flops()

  # from tensorflow.python.keras.utils import plot_model

  # plot_model(mod.model,
  #           to_file='runet.jpg',
  #           #show_shapes=True
  #           )
