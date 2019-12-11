# -*- coding: utf-8 -*-
"""Generator

  File: 
    /hat/dataset/util/generator

  Description: 
    数据迭代器
"""


# import setting
__all__ = [
  'DataGenerator',
  'DG'
]


import gzip
import math
import os
import pickle

import numpy as np
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator # pylint: disable=import-error
from tensorflow.compat.v1.keras.utils import Sequence

from hat.util import log


class DataGenerator(Sequence):
  """
    Data Generator
  """
  def __init__(self,
    x: np.array,
    y: np.array,
    batch_size: int,
    aug: ImageDataGenerator=None,
    **kwargs):
    self.store_x = np.array(x)
    self.store_y = np.array(y)
    self.batch_size = batch_size
    self.length = len(x)
    self.aug = aug

    self.inx = 0
    # self.x = []
    # self.y = []
    log.info(f"Data Generator is Ready", name=__name__)

  def __len__(self):
    return math.ceil(self.length / self.batch_size)

  def __getitem__(self, idx):

    if self.inx == self.__len__():
      self.inx = 0

    if self.inx + 1 == self.__len__():
      batch_x = self.store_x[self.inx * self.batch_size:]
      batch_y = self.store_y[self.inx * self.batch_size:]
    else:
      batch_x = self.store_x[self.inx * self.batch_size:(self.inx + 1) * self.batch_size]
      batch_y = self.store_y[self.inx * self.batch_size:(self.inx + 1) * self.batch_size]
    
    self.inx += 1

    if self.aug:
      batch_x = next(self.aug.flow(batch_x, batch_size=self.batch_size))

    return batch_x, batch_y


class PathDataGenerator(Sequence):
  """
    Path Data Generator
  """
  def __init__(self,
    path: str,
    mode: str,
    batch_size: int,
    data_len: int,
    aug: ImageDataGenerator=None,
    suffix='.gz',
    **kwargs):
    self.path = path
    self.mode = mode
    self.batch_size = batch_size
    self.data_len = data_len
    self.aug = aug
    self.suffix = suffix

    self.x = []
    self.y = []
    self.inx = 0

  def __len__(self):
    return math.ceil(self.data_len / self.batch_size)

  def __getitem__(self, idx):

    next_pkl = os.path.join(self.path, f'{self.mode}{self.inx}{self.suffix}')

    if len(self.x) < self.batch_size and not os.path.exists(next_pkl):
      batch_x = np.array(self.x)
      batch_y = np.array(self.y)
      self.x = []
      self.y = []
      self.inx = 0
    else:
      if len(self.x) < self.batch_size:
        with gzip.open(next_pkl, 'rb') as f:
          pack = pickle.load(f)
          self.x.extend(pack[f'{self.mode}_x'])
          self.y.extend(pack[f'{self.mode}_y'])
          self.inx += 1
      batch_x = np.array(self.x[:self.batch_size])
      batch_y = np.array(self.y[:self.batch_size])
      self.x = self.x[self.batch_size:]
      self.y = self.y[self.batch_size:]

    if self.aug:
      batch_x = next(self.aug.flow(batch_x, batch_size=self.batch_size))

    return batch_x, batch_y


# mini-name
DG = DataGenerator


# test part
if __name__ == "__main__":
  from hat.utils._TC import _TC
  from hat.dataset.lib.mnist import mnist
  t = _TC()
  data = mnist(t)
  # d = DG()
  # print(data.train_x.shape)
  d = DG(data.train_x, data.train_y, 11000)
  for i in range(d.__len__()):
    print(d.__getitem__(0)[0].shape)

