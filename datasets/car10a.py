

from hat.datasets.Dataset import Dataset
from hat.datasets.utils import DSBuilder

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
    self.INPUT_SHAPE = (256, 256, 3)
    self.DATA_DIR = 'datasets/car10a'
    
    self.dsb = DSBuilder(
      self.DATA_DIR,
      self.INPUT_SHAPE[0:2],
      shuffle=self.SHUFFLE)
    self.CLASSES_DICT = self.dsb.get_classes_dict()
    (self.train_x, self.train_y), (self.val_x, self.val_y), self.test_x = self.dsb.get_all('fillx')


if __name__ == "__main__":
  from pprint import pprint
  car10a = car10a()
  pprint(car10a.CLASSES_DICT)
