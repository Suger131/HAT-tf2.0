from .Dataset import Dataset
import tensorflow.keras.datasets as ds


class cifar10(Dataset):
  """
  cifar10 数据集
  """

  def args(self):
    self._MISSION_LIST = ['classfication']
    self.NUM_TRAIN = 50000
    self.NUM_TEST = 10000
    self.NUM_CLASSES = 10
    self.IMAGE_SHAPE = (32, 32, 3)
    (self.train_images, self.train_labels), (self.test_images, self.test_labels) = ds.cifar10.load_data()
    self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0
