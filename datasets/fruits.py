from hat.datasets.utils import Dataset
from hat.datasets.utils import DSBuilder


# import setting
__all__ = [
  'fruits',
  'fruitsb',
]


class fruits(Dataset):
  """
    Fruits 数据集
  """
  
  def args(self):
    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 58266
    self.NUM_VAL = 19548
    self.NUM_TEST = 0
    self.NUM_CLASSES = 114
    self.INPUT_SHAPE = (100, 100, 3)
    self.DATA_DIR = 'E:/1-ML/Fruits114'
    self.PKL_DIR = f'{self.DATA_DIR}/fruits'

    self.dsb = DSBuilder(
      self.DATA_DIR,
      self.INPUT_SHAPE[0:2],
      pkldir=self.PKL_DIR,
      shuffle=self.SHUFFLE)
    self.CLASSES_DICT = self.dsb.get_classes_dict()
    (self.train_x, self.train_y), (self.val_x, self.val_y), self.test_x = self.dsb.get_all('ignore')
    self.train_x = self.train_x / 255.0
    self.val_x = self.val_x / 255.0
    self.train_x = self.train_x.astype(self.dtype)
    self.val_x = self.val_x.astype(self.dtype)


class fruitsb(Dataset):
  """
    Fruitsb 数据集
  """
  
  def args(self):
    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 58266
    self.NUM_VAL = 19548
    self.NUM_TEST = 0
    self.NUM_CLASSES = 114
    self.INPUT_SHAPE = (224, 224, 3)
    self.DATA_DIR = 'E:/1-ML/Fruits114'
    self.PKL_DIR = f'{self.DATA_DIR}/fruitsb'

    self.dsb = DSBuilder(
      self.DATA_DIR,
      self.INPUT_SHAPE[0:2],
      pkldir=self.PKL_DIR,
      shuffle=self.SHUFFLE)
    self.CLASSES_DICT = self.dsb.get_classes_dict()
    (self.train_x, self.train_y), (self.val_x, self.val_y), self.test_x = self.dsb.get_all('stretch')
    self.train_x = self.train_x / 255.0
    self.val_x = self.val_x / 255.0
    self.train_x = self.train_x.astype(self.dtype)
    self.val_x = self.val_x.astype(self.dtype)


# test part
if __name__ == "__main__":
  from pprint import pprint
  data = fruitsb()
  pprint(data.CLASSES_DICT)
  print(data.train_x.shape)
  # from PIL import Image
  # Image.fromarray(data.train_x[5]).show()
