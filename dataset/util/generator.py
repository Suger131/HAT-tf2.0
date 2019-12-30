# -*- coding: utf-8 -*-
"""Generator

  File: 
    /hat/dataset/util/generator

  Description: 
    数据迭代器
"""


# import setting
__all__ = [
    'Generator',]


import gzip
import math
import os
import pickle

import numpy as np
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator # pylint: disable=import-error
from tensorflow.keras.utils import Sequence

from hat.util import log


class Generator(object):
  """Generator

    Description: 
      数据集的生成器工具

    Args:
      x: np.array. 推荐使用`hat.Dataset.train_x`.
      y: np.array. 推荐使用`hat.Dataset.train_y`.
      batch_size: Int. 批量大小.
      aug: keras.ImageDataGenerator, default None. 数据增强器
      **kwargs: Any. 任意参数，自动忽略

    Usage:
    ```python
      from hat.dataset.lib.mnist import mnist
      data = mnist()
      g = Generator(data.train_x, data.train_y, 11000)
      for batch in g:  # epoch mode
        pass  # todo
      for inx in range(step):  # step mode
        batch = next(g)
        pass  # todo
    ```
  """
  def __init__(
      self,
      x: np.array,
      y: np.array,
      batch_size: int,
      shuffle=True,
      aug=None,
      **kwargs):
    self.x = x
    self.y = y
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.aug = aug
    self.len = math.ceil(len(x) / self.batch_size)
    self.iterator = self.__iterator__()
  
  def __next__(self):
    return next(self.iterator)

  def __iter__(self):
    for i in range(self.len):
      yield next(self)

  def __del__(self):
    try:
      del self.iterator
    except Exception:
      pass

  def __iterator__(self):
    while True:
      epoch = self._shuffle([self.x, self.y])
      for inx in range(self.len):
        if inx + 1 == self.len:
          x = epoch[0][inx * self.batch_size:]
          y = epoch[1][inx * self.batch_size:]
        else:
          x = epoch[0][inx * self.batch_size:(inx + 1) * self.batch_size]
          y = epoch[1][inx * self.batch_size:(inx + 1) * self.batch_size]
        if self.aug is not None:
          x = next(self.aug.flow(x, batch_size=self.batch_size))
        yield x, y

  def _shuffle(self, inputs):
    if not self.shuffle:
      return inputs
    r = np.random.permutation(len(inputs[-1]))
    return inputs[0][r], inputs[1][r]


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
    self.len = len(x)
    self.aug = aug

    self.inx = 0
    # self.x = []
    # self.y = []
    log.info(f"Data Generator is Ready", name=__name__)

  def __len__(self):
    return math.ceil(self.len / self.batch_size)

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

