# pylint: disable=attribute-defined-outside-init

from hat.datasets.Dataset import Dataset
from hat.datasets.utils import DSBuilder


class dogx(Dataset):
  """
    dogx 数据集
  """
  
  def args(self):
    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 18180
    self.NUM_VAL = 2400
    self.NUM_TEST = 0
    self.NUM_CLASSES = 120
    self.INPUT_SHAPE = (64, 64, 3)
    self.DATA_DIR = 'datasets/dogx'

    self.dsb = DSBuilder(
      self.DATA_DIR,
      self.INPUT_SHAPE[0:2],
      shuffle=self.SHUFFLE)
    self.CLASSES_DICT = self.dsb.get_classes_dict()
    (self.train_x, self.train_y), (self.val_x, self.val_y), self.test_x = self.dsb.get_all('stretch')


# test part
if __name__ == "__main__":
  from pprint import pprint
  data = dogx()
  pprint(data.CLASSES_DICT)
  print(data.train_x.shape)
