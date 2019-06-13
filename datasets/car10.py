import os
import pickle
import gzip
from random import shuffle
import numpy as np
from PIL import Image
from tensorflow.python.keras import datasets as ds
from hat.datasets.Dataset import Dataset


class car10(Dataset):
  """
  car10 数据集
  """

  def args(self):
    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 1400
    self.NUM_VAL = 200
    self.NUM_TEST = 200
    self.NUM_CLASSES = 10
    self.INPUT_SHAPE = (300, 300, 3)
    self.DATA_DIR = 'datasets/car10'
    self.CLASSES_DICT = self._get_classes_dict()
    if not self._load():
      self._save()

  def _get_classes_dict(self):
    with open(f"{self.DATA_DIR}/classes.txt", 'r') as f:
      temp = [i.strip() for i in f.readlines()]
    return dict(zip(range(self.NUM_CLASSES), temp))

  def _get_test_data(self):
    images = []
    for i in range(self.NUM_TEST):
      img = self._image_reshape(f"{self.DATA_DIR}/test/{i}.jpg")
      images.append(img)
    images = np.array(images)
    return images

  def _get_val_data(self):
    images = []
    labels = []
    for item in self.CLASSES_DICT:
      for i in range(self.NUM_VAL // self.NUM_CLASSES):
        img = self._image_reshape(f"{self.DATA_DIR}/val/{self.CLASSES_DICT[item]}/{i}.jpg")
        images.append(img)
        labels.append(item)
    images, labels = self._shuffle([images, labels])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

  def _get_train_data(self):
    images = []
    labels = []
    for item in self.CLASSES_DICT:
      for i in range(self.NUM_TRAIN // self.NUM_CLASSES):
        img = self._image_reshape(f"{self.DATA_DIR}/train/{self.CLASSES_DICT[item]}/{i}.jpg")
        images.append(img)
        labels.append(item)
    images, labels = self._shuffle([images, labels])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

  def _image_reshape(self, img_name):
    img = Image.open(img_name)

    w, h = img.size
    if h < self.INPUT_SHAPE[0] and w < self.INPUT_SHAPE[1]:
      pass
    if h >= w:
      _w = int(w / h * self.INPUT_SHAPE[0])
      img = img.resize((_w, self.INPUT_SHAPE[0]))
    if w > h:
      _h = int(h / w * self.INPUT_SHAPE[1])
      img = img.resize((self.INPUT_SHAPE[1], _h))

    w, h = img.size
    p_w = self.INPUT_SHAPE[0] - w
    p_h = self.INPUT_SHAPE[1] - h
    pad_w = (int(p_w / 2), p_w - int(p_w / 2))
    pad_h = (int(p_h / 2), p_h - int(p_h / 2))

    img = np.array(img)
    img = np.pad(img, (pad_h, pad_w, (0, 0)), 'constant', constant_values=0)
    
    return img

  def _shuffle(self, inputs, islist=True):
    if self.SHUFFLE:
      if islist:
        len_ = len(inputs[0])
        index = list(range(len_))
        shuffle(index)
        temp = [[i[index[j]] for j in range(len_)] for i in inputs]
      else:
        len_ = len(inputs)
        index = list(range(len_))
        shuffle(index)
        temp = [inputs[i] for i in index]
      return temp
    return inputs

  def _save(self):
    self.train_x, self.train_y = self._get_train_data()
    self.val_x, self.val_y = self._get_val_data()
    self.test_x = self._get_test_data()
    car10_ = {'train_x': self.train_x,
              'train_y': self.train_y,
              'val_x'  : self.val_x,
              'val_y'  : self.val_y,
              'test_x' : self.test_x}
    with gzip.open(f"{self.DATA_DIR}/car10.gz", 'wb') as f:
      pickle.dump(car10_, f)

  def _load(self):
    if not os.path.exists(f"{self.DATA_DIR}/car10.gz"):
      return False
    with gzip.open(f"{self.DATA_DIR}/car10.gz",'rb') as f:
      car10_ = pickle.load(f)
    self.train_x = car10_['train_x']
    self.train_y = car10_['train_y']
    self.val_x   = car10_['val_x']
    self.val_y   = car10_['val_y']
    self.test_x = car10_['test_x']
    return True

# test part
if __name__ == "__main__":
  from pprint import pprint
  car10 = car10()
  pprint(car10.CLASSES_DICT)
