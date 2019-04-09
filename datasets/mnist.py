from .Packer import Packer
import tensorflow.keras.datasets as ds


class mnist(Packer):

  def __init__(self):
    super(mnist, self).__init__()
    self.NUM_TRAIN = 60000
    self.NUM_TEST = 10000
    self.NUM_CLASSES = 10
    self.IMAGE_SIZE = 28
    self.IMAGE_DEPTH = 1
    (self.train_images, self.train_labels), (self.test_images, self.test_labels) = ds.mnist.load_data()
    self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0
    self.train_images = self.train_images.reshape((self.NUM_TRAIN, self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_DEPTH))
    self.test_images = self.test_images.reshape((self.NUM_TEST, self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_DEPTH))

# test mode
if __name__ == "__main__":
  class Args:
    pass
  a = Args()
  m = mnist()
  m.ginfo(a)
  print(a.__dict__)
