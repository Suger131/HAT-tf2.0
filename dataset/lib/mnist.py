# -*- coding: utf-8 -*-
"""Mnist

  File: 
    /hat/dataset/lib/mnist

  Description: 
    mnist系列数据集，包含
    1. mnist
    2. fashion_mnist(f_mnist)
"""


# import setting
__all__ = [
    'mnist',
    'fashion_mnist',
    'f_mnist',]


import tensorflow as tf

from hat.dataset.util import dataset


class mnist(dataset.Dataset):
  """
    mnist 数据集
  """
  def args(self):
    self.mission_list = ['classfication']
    self.num_train = 60000
    self.num_val = 10000
    self.num_test = 0
    self.input_shape = (28, 28, 1)
    self.output_shape = (10,)
    (self.train_x, self.train_y), (self.val_x, self.val_y) = tf.keras.datasets.mnist.load_data()
    self.test_x = None
    self.test_y = None

    self.train_x = self.train_x / 255.0
    self.train_x = self.train_x.reshape((self.num_train, *self.input_shape))
    self.val_x = self.val_x / 255.0
    self.val_x = self.val_x.reshape((self.num_val, *self.input_shape))


class fashion_mnist(dataset.Dataset):
  """
    fashion_mnist 数据集
  """
  def args(self):
    self.mission_list = ['classfication']
    self.num_train = 60000
    self.num_val = 10000
    self.num_test = 0
    self.input_shape = (28, 28, 1)
    self.output_shape = (10,)
    (self.train_x, self.train_y), (self.val_x, self.val_y) = tf.keras.datasets.fashion_mnist.load_data()
    self.test_x = None
    self.test_y = None
    
    self.train_x = self.train_x / 255.0
    self.train_x = self.train_x.reshape((self.num_train, *self.input_shape))
    self.val_x = self.val_x / 255.0
    self.val_x = self.val_x.reshape((self.num_val, *self.input_shape))


# Alias
f_mnist = fashion_mnist


# test
if __name__ == "__main__":
  from hat.utils._TC import _TC
  t = _TC()
  d = mnist(t)
  print(t.train_x.shape)

