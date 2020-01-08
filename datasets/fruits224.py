# pylint: disable=attribute-defined-outside-init

from hat.datasets.Dataset import Dataset
from hat.datasets.utils import DSBuilder


class fruits224(Dataset):
  """
    Fruits224 数据集
  """
  
  def args(self):
    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 58266
    self.NUM_VAL = 19548
    self.NUM_TEST = 0
    self.NUM_CLASSES = 114
    self.INPUT_SHAPE = (224, 224, 3)
    self.DATA_DIR = 'datasets/fruits224'

    self.dsb = DSBuilder(
      self.DATA_DIR,
      self.INPUT_SHAPE[0:2],
      shuffle=self.SHUFFLE)
    self.CLASSES_DICT = self.dsb.get_classes_dict()
    (self.train_x, self.train_y), (self.val_x, self.val_y), self.test_x = self.dsb.get_all('stretch')


# test part
if __name__ == "__main__":
  from pprint import pprint
  data = fruits224()
  pprint(data.CLASSES_DICT)
  print(data.train_x.shape)
  from PIL import Image
  Image.fromarray(data.train_x[5]).show()
