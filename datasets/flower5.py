# pylint: disable=attribute-defined-outside-init

from hat.datasets.Dataset import Dataset
from hat.datasets.utils import DSBuilder


class flower5(Dataset):
  """
    Flower5 数据集
  """
  
  def args(self):
    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 3603
    self.NUM_VAL = 720
    self.NUM_TEST = 0
    self.NUM_CLASSES = 5
    self.INPUT_SHAPE = (256, 256, 3)
    self.DATA_DIR = 'datasets/flower5'

    self.dsb = DSBuilder(
      self.DATA_DIR,
      self.INPUT_SHAPE[0:2],
      shuffle=self.SHUFFLE)
    self.CLASSES_DICT = self.dsb.get_classes_dict()
    (self.train_x, self.train_y), (self.val_x, self.val_y), self.test_x = self.dsb.get_all('fillx')


# test part
if __name__ == "__main__":
  from pprint import pprint
  data = flower5()
  pprint(data.CLASSES_DICT)
  print(data.train_x.shape)
  from PIL import Image
  Image.fromarray(data.train_x[0]).show()