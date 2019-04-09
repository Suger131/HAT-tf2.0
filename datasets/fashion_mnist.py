from .Packer import Packer
import tensorflow.keras.datasets as ds


class fashion_mnist(Packer):

  def __init__(self):
    super(fashion_mnist, self).__init__()
    self.NUM_TRAIN = 60000
    self.NUM_TEST = 10000
    self.NUM_CLASSES = 10
    self.IMAGE_SIZE = 28
    self.IMAGE_DEPTH = 1
    (self.train_images, self.train_labels), (self.test_images, self.test_labels) = ds.fashion_mnist.load_data()
    self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0
    self.train_images = self.train_images.reshape((self.NUM_TRAIN, self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_DEPTH))
    self.test_images = self.test_images.reshape((self.NUM_TEST, self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_DEPTH))
    