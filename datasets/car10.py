# pylint: disable=attribute-defined-outside-init

from hat.datasets.utils import Dataset
from hat.datasets.utils import DSBuilder


# import setting
__all__ = [
  'car10',
  'car10a',
  'car10m',
]


class car10(Dataset):
  """
    Car10 数据集
  """

  def args(self):
    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 1400
    self.NUM_VAL = 200
    self.NUM_TEST = 200
    self.NUM_CLASSES = 10
    self.INPUT_SHAPE = (224, 224, 3)
    self.DATA_DIR = 'E:/1-ML/Car10'
    self.PKL_DIR = f'{self.DATA_DIR}/car10'

    self.dsb = DSBuilder(
      self.DATA_DIR,
      self.INPUT_SHAPE[0:2],
      pkldir=self.PKL_DIR,
      shuffle=self.SHUFFLE)
    self.CLASSES_DICT = self.dsb.get_classes_dict()
    (self.train_x, self.train_y), (self.val_x, self.val_y), self.test_x = self.dsb.get_all('imagenet')
    self.train_x, self.val_x = self.train_x / 255.0, self.val_x / 255.0
    self.train_x = self.train_x.astype(self.dtype)
    self.val_x = self.val_x.astype(self.dtype)


class car10a(Dataset):
  """
    Car10a 数据集
  """
  
  def args(self):
    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 1400
    self.NUM_VAL = 200
    self.NUM_TEST = 200
    self.NUM_CLASSES = 10
    self.INPUT_SHAPE = (224, 224, 3)
    self.DATA_DIR = 'E:/1-ML/Car10'
    self.PKL_DIR = f'{self.DATA_DIR}/car10a'
    
    self.dsb = DSBuilder(
      self.DATA_DIR,
      self.INPUT_SHAPE[0:2],
      pkldir=self.PKL_DIR,
      shuffle=self.SHUFFLE)
    self.CLASSES_DICT = self.dsb.get_classes_dict()
    (self.train_x, self.train_y), (self.val_x, self.val_y), self.test_x = self.dsb.get_all('fillx')
    self.train_x, self.val_x = self.train_x / 255.0, self.val_x / 255.0
    self.train_x = self.train_x.astype(self.dtype)
    self.val_x = self.val_x.astype(self.dtype)


class car10m(Dataset):
  """
    Car10m 数据集
  """

  def args(self):
    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 1400
    self.NUM_VAL = 200
    self.NUM_TEST = 200
    self.NUM_CLASSES = 10
    self.INPUT_SHAPE = (32, 32, 3)
    self.DATA_DIR = 'E:/1-ML/Car10'
    self.PKL_DIR = f'{self.DATA_DIR}/car10m'

    self.dsb = DSBuilder(
      self.DATA_DIR,
      self.INPUT_SHAPE[0:2],
      pkldir=self.PKL_DIR,
      shuffle=self.SHUFFLE)
    self.CLASSES_DICT = self.dsb.get_classes_dict()
    (self.train_x, self.train_y), (self.val_x, self.val_y), self.test_x = self.dsb.get_all('imagenet')
    self.train_x, self.val_x = self.train_x / 255.0, self.val_x / 255.0
    self.train_x = self.train_x.astype(self.dtype)
    self.val_x = self.val_x.astype(self.dtype)


# test part
if __name__ == "__main__":
  from pprint import pprint
  car10 = car10m()
  pprint(car10.CLASSES_DICT)
