from hat.datasets.Dataset import Dataset
from tensorflow.python.keras import datasets as ds


class boston_housing(Dataset):
  """
  boston_housing 数据集
  """

  def args(self):
    self._MISSION_LIST = ['regression', 'classfication']
    self.NUM_TRAIN = 404
    self.NUM_TEST = 102
    self.NUM_CLASSES = 1
    self.INPUT_SHAPE = (13,)
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.boston_housing.load_data()


# test mode
if __name__ == "__main__":
  m = boston_housing()
  # print(m.ginfo())
  print(m.train_x.dtype, m.train_y.dtype)
