"""
  hat.dataset.lib.mnist

  mnist 数据集
  fashion_mnist 数据集
"""

# pylint: disable=no-name-in-module
# pylint: disable=import-error
# pylint: disable=no-member


__all__ = [
  'mnist',
  'fashion_mnist',
  'f_mnist',
]


from hat.dataset.utils.Dataset import Dataset
from tensorflow.python.keras import datasets as ds


class mnist(Dataset):
  """
    mnist 数据集
  """
  def args(self):
    self.mission_list = ['classfication']
    self.num_train = 60000
    self.num_val = 10000
    self.num_test = 0
    self.input_shape = (28, 28, 1)
    self.output_shape = (10,)
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.mnist.load_data()
    self.test_x = None
    self.test_y = None

    self.train_x = self.train_x / 255.0
    self.train_x = self.train_x.reshape((self.num_train, *self.input_shape))
    self.val_x = self.val_x / 255.0
    self.val_x = self.val_x.reshape((self.num_val, *self.input_shape))


class fashion_mnist(Dataset):
  """
    fashion_mnist 数据集
  """
  def args(self):
    self.mission_list = ['classfication']
    self.num_train = 60000
    self.num_val = 10000
    self.num_test = 0
    self.input_shape = (28, 28, 1)
    self.output_shape = (10,)
    (self.train_x, self.train_y), (self.val_x, self.val_y) = ds.fashion_mnist.load_data()
    self.test_x = None
    self.test_y = None
    
    self.train_x = self.train_x / 255.0
    self.train_x = self.train_x.reshape((self.num_train, *self.input_shape))
    self.val_x = self.val_x / 255.0
    self.val_x = self.val_x.reshape((self.num_val, *self.input_shape))


# Alias
f_mnist = fashion_mnist


# test
if __name__ == "__main__":
  from hat.utils._TC import _TC
  t = _TC()
  d = mnist(t)
  print(t.train_x.shape)

