# pylint: disable=attribute-defined-outside-init
# pylint: disable=no-name-in-module

import gzip
import math
import os
import pickle

import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import Sequence


class DG(Sequence):
  """
    Data Generator
  """
  def __init__(self, path: str, mode: str, batch_size: int, data_len: int,
        dtype=None, aug: ImageDataGenerator=None, suffix='.gz'):
    self.path = path
    self.mode = mode
    self.batch_size = batch_size
    self.data_len = data_len
    self.dtype = dtype or 'float32'
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

    batch_x = batch_x / 255.0
    batch_x = batch_x.astype(self.dtype)
    
    return batch_x, batch_y


if __name__ == "__main__":
  AUG = ImageDataGenerator(
      rotation_range=30,
      width_shift_range=0.05,
      height_shift_range=0.05,
      shear_range=0.05,
      zoom_range=0.05,
      horizontal_flip=True,
  )
  INDG = DG('E:/1-ML/ImageNet/pkl', 'val', 128, 10000)#, AUG
  print(len(INDG))
  for x, y in INDG:
    print(len(x), len(y))
  
