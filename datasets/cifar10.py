from datasets.Dataset import Dataset
from tensorflow.python.keras import datasets as ds


class cifar10(Dataset):
  """
  cifar10 数据集
  """

  def args(self):
    self._MISSION_LIST = ['classfication']
    self.NUM_TRAIN = 50000
    self.NUM_TEST = 10000
    self.NUM_CLASSES = 10
    self.INPUT_SHAPE = (32, 32, 3)
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.cifar10.load_data()
    self.train_x, self.val_x = self.train_x / 255.0, self.val_x / 255.0
