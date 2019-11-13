"""
  hat.dataset.utils.DataGenerator

  数据生成器
"""

# pylint: disable=unused-argument
# pylint: disable=attribute-defined-outside-init
# pylint: disable=no-name-in-module
# pylint: disable=import-error


__all__ = [
  'DataGenerator',
  'DG'
]


import gzip
import math
import os
import pickle

import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import Sequence


class DataGenerator(Sequence):
  """
    Data Generator
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

