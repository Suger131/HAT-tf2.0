from hat.datasets.Dataset import Dataset
from tensorflow.python.keras import datasets as ds


class imagenet(Dataset):
  """
    ImageNet 数据集
  """

  def args(self):
    self._MISSION_LIST = ['classfication']
    self.NUM_TRAIN = 60000
    self.NUM_TEST = 10000
    self.NUM_VAL = 10000
    self.NUM_CLASSES = 10
    self.INPUT_SHAPE = (224, 224, 3)
    (self.train_x, self.train_y), (self.val_x, self.val_y) = (None, None), (None, None)


# test mode
if __name__ == "__main__":
  m = imagenet()
  print(m.ginfo())
#   print(m.train_x.dtype)
