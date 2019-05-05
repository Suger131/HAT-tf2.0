'''
  VGG19 模型
  本模型默认总参数量[参考基准：cifar10]：
    Total params:           42,592,962
    Trainable params:       42,582,210
    Non-trainable params:   10,752
  本模型默认总参数量[参考基准：ImageNet]：
    Total params:           143,256,258
    Trainable params:       143,245,506
    Non-trainable params:   10,752
'''


from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras.models import Model


class vgg19(NetWork, AdvNet):
  """
  VGG19 模型
  """

  def args(self):
    self.NUM_LAYERS = [2, 3, 3, 4, 4]
    self.CONV = [64, 128, 256, 512, 512]

    self.LOCAL = [4096, 4096, 1000]
    self.DROP = 0.3

    # for test
    # self.BATCH_SIZE = 128
    # self.EPOCHS = 150
    # self.OPT = 'sgd'
    # self.OPT_EXIST = True

  def build_model(self):
    x_in = self.input(self.IMAGE_SHAPE)

    # 卷积部分
    x = x_in
    x = self.repeat(self.conv_bn, self.NUM_LAYERS[0], x, self.CONV[0], 3)
    x = self.maxpool(x)
    x = self.repeat(self.conv_bn, self.NUM_LAYERS[1], x, self.CONV[1], 3)
    x = self.maxpool(x)
    x = self.repeat(self.conv_bn, self.NUM_LAYERS[2], x, self.CONV[2], 3)
    x = self.maxpool(x)
    x = self.repeat(self.conv_bn, self.NUM_LAYERS[3], x, self.CONV[3], 3)
    x = self.maxpool(x, 2 if self.IMAGE_SHAPE[0] // 16 >= 4 else 1)
    x = self.repeat(self.conv_bn, self.NUM_LAYERS[4], x, self.CONV[4], 3) 
    x = self.maxpool(x) if self.IMAGE_SHAPE[0] // 32 >= 4 else self.GAPool(x)

    # 全连接部分
    x = self.flatten(x)
    x = self.local(x, self.LOCAL[0])
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.LOCAL[1])
    x = self.dropout(x, self.DROP)
    x = self.local(x, self.LOCAL[2])
    x = self.local(x, self.NUM_CLASSES, activation='softmax')

    self.model = Model(inputs=x_in, outputs=x, name='vgg19')

# test part
if __name__ == "__main__":
  mod = vgg19(DATAINFO={'IMAGE_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  print(mod.IMAGE_SHAPE)
  print(mod.model.summary())
