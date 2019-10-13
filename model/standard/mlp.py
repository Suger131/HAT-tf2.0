"""
  默认模型
  简单三层神经网络[添加了Dropout层]
"""


import tensorflow as tf
from hat.model.utils.network_v1 import Network_v1
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *


class mlp(Network_v1):
  """
    MLP
  """
  def args(self):
    self.node = 128
    self.drop = .5

  def build(self):
    x_in = tf.keras.layers.Input(self.config.input_shape)
    x = x_in
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(self.drop)(x)
    x = tf.keras.layers.Dense(self.config.output_shape[0], activation='softmax')(x)
    return Model(x_in, x, name='mlp')


# test
if __name__ == "__main__":
  from hat.utils.tconfig import tconfig
  t = tconfig()
  t.input_shape = (28, 28, 1)
  t.output_shape = (10,)
  mod = mlp(t)
  t.model.summary()
  # mod.flops()
