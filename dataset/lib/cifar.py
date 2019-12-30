# -*- coding: utf-8 -*-
"""Cifar

  File: 
    /hat/dataset/lib/cifar

  Description: 
    cifar系列数据集，包含
    1. Cifar-10
    2. Cifar-100
"""


# import setting
__all__ = [
    'cifar10',
    'cifar100',]


import tensorflow as tf

from hat.dataset import Dataset


class cifar10(Dataset):
  """Cifar-10
    
    Description:
      None
  """
  def args(self):
    self.mission_list = ['classfication']
    self.num_train = 50000
    self.num_val = 10000
    self.num_test = 0
    self.input_shape = (32, 32, 3)
    self.output_shape = (10,)
    (self.train_x, self.train_y), (self.val_x, self.val_y) = tf.keras.datasets.cifar10.load_data()
    self.test_x = None
    self.test_y = None
    self.train_x = self.train_x / 255.0
    self.val_x = self.val_x / 255.0


class cifar100(Dataset):
  """Cifar-100
    
    Description:
      None
  """
  def args(self):
    self.mission_list = ['classfication']
    self.num_train = 50000
    self.num_val = 10000
    self.num_test = 0
    self.input_shape = (32, 32, 3)
    self.output_shape = (100,)
    (self.train_x, self.train_y), (self.val_x, self.val_y) = tf.keras.datasets.cifar100.load_data()
    self.test_x = None
    self.test_y = None
    self.train_x = self.train_x / 255.0
    self.val_x = self.val_x / 255.0


# test
if __name__ == "__main__":
  data = cifar10()
  print(data.train_x.shape)

