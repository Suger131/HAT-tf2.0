'''
  默认模型
  For regression
  简单三层神经网络[添加了Dropout层]
  本模型默认总参数量[参考基准：boston-housing]：
    Total params:           3,082
    Trainable params:       3,082
    Non-trainable params:   0
'''

from models.network import NetWork
from models.advanced import AdvNet
from tensorflow.python.keras.models import Model


class mlp_r(NetWork, AdvNet):
  """
  MLP-R 模型
  """
  
  def args(self):
    self.LOCAL_SIZE = 128
    self.DROP_RATE = 0.5

    # for test
    # self.BATCH_SIZE = 128
    # self.EPOCHS = 150
    self.OPT = 'rmsprop'
    self.OPT_EXIST = True
    self.LOSS_MODE = 'mse'
    self.METRICS = ['mae']

  def build_model(self):
    x_in = self.input(self.INPUT_SHAPE)

    x = self.flatten(x_in)
    x = self.local(x, self.LOCAL_SIZE)
    x = self.dropout(x, self.DROP_RATE)
    x = self.local(x, self.NUM_CLASSES)

    self.model = Model(inputs=x_in, outputs=x, name='mlp_r')


# test part
if __name__ == "__main__":
  mod = mlp_r(DATAINFO={'INPUT_SHAPE': (13,), 'NUM_CLASSES': 10})
  print(mod.INPUT_SHAPE)
  print(mod.model.summary())
