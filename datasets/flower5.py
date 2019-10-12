from hat.datasets.utils import Dataset
from hat.datasets.utils import DSBuilder


# import setting
__all__ = [
  'flower5',
  'flower5m',
]


class flower5(Dataset):
  """
    Flower5 数据集
  """
  
  def args(self):
    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 3498
    self.NUM_VAL = 700
    self.NUM_TEST = 0
    self.NUM_CLASSES = 5
    self.INPUT_SHAPE = (224, 224, 3)
    self.DATA_DIR = 'E:/1-ML/Flower5'
    self.PKL_DIR = f'{self.DATA_DIR}/flower5'

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


class flower5m(Dataset):
  """
    Flower5m 数据集
  """
  
  def args(self):
    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 3498
    self.NUM_VAL = 700
    self.NUM_TEST = 0
    self.NUM_CLASSES = 5
    self.INPUT_SHAPE = (50, 50, 3)
    self.DATA_DIR = 'E:/1-ML/Flower5'
    self.PKL_DIR = f'{self.DATA_DIR}/flower5m'

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
  data = flower5m()
  pprint(data.CLASSES_DICT)
  print(data.train_x.shape)
  # from PIL import Image
  # imgl = [Image.fromarray(data.train_x[i]).save(f'{data.DATA_DIR}/{i}.jpg') for i in range(40)]