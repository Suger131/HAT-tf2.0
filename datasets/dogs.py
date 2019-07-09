# pylint: disable=attribute-defined-outside-init

import os
import pickle
import gzip
from random import shuffle
import numpy as np
from PIL import Image
from hat.datasets.Dataset import Dataset


class dogs(Dataset):
  """
    dogs 数据集
  """
  
  def args(self):
    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 19380
    self.NUM_VAL = 1200
    self.NUM_TEST = 0
    self.NUM_CLASSES = 120
    self.INPUT_SHAPE = (300, 300, 3)
    self.PKL_NUM = 4096
    self.PKL_SIZE = self.INPUT_SHAPE[0] * self.INPUT_SHAPE[1] * self.INPUT_SHAPE[2] * self.PKL_NUM
    self.DATA_DIR = 'datasets/dogs'
    self.DATA_PKL = self.DATA_DIR + '/filelist.txt'
    self.CLASSES_DICT = self._get_classes_dict()
    if not self._load():
      self._save()

  def _get_classes_dict(self):
    with open(f"{self.DATA_DIR}/classes.txt", 'r') as f:
      temp = [i.strip() for i in f.readlines()]
    return dict(zip(range(self.NUM_CLASSES), temp))

  def _get_data(self, name):
    if name != 'test':
      images = []
      labels = []
      for item in self.CLASSES_DICT:
        _dir = f"{self.DATA_DIR}/{name}/{self.CLASSES_DICT[item]}/"
        count = 0
        for files in os.listdir(_dir):
          count += 1
        for i in range(count):
          img = self._image_reshape(_dir + f'{i}.jpg')
          images.append(img)
          labels.append(item)
      images, labels = self._shuffle([images, labels])
      images = np.array(images)
      labels = np.array(labels)
      return images, labels
    else:
      images = []
      _dir = f"{self.DATA_DIR}/{name}/"
      for files in os.listdir(_dir):
        img = self._image_reshape(_dir + files)
        images.append(img)
      images = np.array(images)
      return images

  def _image_reshape(self, img_name):
    img = Image.open(img_name)

    if img.mode != 'RGB':
      print(img_name, img.mode)
      img.convert('RGB')

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
    self.train_x, self.train_y = self._get_data('train')
    self.val_x, self.val_y = self._get_data('val')
    # self.test_x = self._get_data('test')
    
    file_list = []

    for i in range((self.NUM_TRAIN + self.PKL_NUM - 1) // self.PKL_NUM):
      if (i + 1) * self.PKL_NUM <= self.NUM_TRAIN:
        train = {
          'train_x': self.train_x[i*self.PKL_NUM:(i + 1)*self.PKL_NUM],
          'train_y': self.train_y[i*self.PKL_NUM:(i + 1)*self.PKL_NUM],
        }
      else:
        train = {
          'train_x': self.train_x[i*self.PKL_NUM:],
          'train_y': self.train_y[i*self.PKL_NUM:],
        }
      with gzip.open(f"{self.DATA_DIR}/train{i}.gz", 'wb') as f:
        pickle.dump(train, f)
      file_list.append(f'train{i}.gz\n')

    val = {
      'val_x': self.val_x,
      'val_y': self.val_y,
    }
    with gzip.open(self.DATA_DIR + '/val.gz', 'wb') as f:
      pickle.dump(val, f)
    file_list.append('val.gz\n')

    with open(self.DATA_PKL, 'w') as f:
      f.writelines(file_list)

  def _load(self):
    if not os.path.exists(self.DATA_PKL):
      return False
    
    with open(self.DATA_PKL, 'r') as f:
      filelist = [i.strip() for i in f.readlines()]

    train_list = [i for i in filelist if 'train' in i]
    train_x, train_y = [], []
    for ix, i in enumerate(train_list):
      with gzip.open(f"{self.DATA_DIR}/{i}", 'rb') as f:
        train = pickle.load(f)
        train_x.append(train['train_x'])
        train_y.append(train['train_y'])
    self.train_x = np.concatenate(train_x)
    self.train_y = np.concatenate(train_y)
    del train_x, train_y

    val_list = [i for i in filelist if 'val' in i]
    val_x, val_y = [], []
    for ix, i in enumerate(val_list):
      with gzip.open(f"{self.DATA_DIR}/{i}", 'rb') as f:
        val = pickle.load(f)
        val_x.append(val['val_x'])
        val_y.append(val['val_y'])
    self.val_x = np.concatenate(val_x)
    self.val_y = np.concatenate(val_y)
    del val_x, val_y

    return True


# test part
if __name__ == "__main__":
  from pprint import pprint
  data = dogs()
  pprint(data.CLASSES_DICT)
