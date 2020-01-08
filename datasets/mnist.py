from hat.datasets.Dataset import Dataset
from tensorflow.python.keras import datasets as ds


class mnist(Dataset):
  """
    mnist 数据集
  """

  def args(self):
    self._MISSION_LIST = ['classfication']
    self.NUM_TRAIN = 60000
    self.NUM_TEST = 10000
    self.NUM_CLASSES = 10
    self.INPUT_SHAPE = (28, 28, 1)
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.mnist.load_data()
    self.train_x, self.val_x = self.train_x / 255.0, self.val_x / 255.0
    self.train_x = self.train_x.reshape((self.NUM_TRAIN, *self.INPUT_SHAPE))
    self.val_x = self.val_x.reshape((self.NUM_TEST, *self.INPUT_SHAPE))


# test mode
if __name__ == "__main__":
  m = mnist()
  print(m.ginfo())
  print(m.train_x.dtype)
