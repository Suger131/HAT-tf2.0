
import gzip
import os
import pickle

import numpy as np
from PIL import Image


__all__ = [
  '_shuffle',
  'DSBuilder'
]


def _shuffle(inputs, islist=True):
  """
    Shuffle the datas/labels

    Argu:
      inputs: np.array/list, or list/tuple of np.array/list. 
      In the latter case, the corresponding elements for each 
      item in the list are shuffled together as a whole.

      islist: Boolean. The latter case is enabled when the 
      value is True. Default is True.

    Return:
      A shuffled np.array/list(depend on argus)
  """
  from random import shuffle
  if islist:
    len_ = len(inputs[0])
    index = list(range(len_))
    shuffle(index)
    outputs = [[i[index[j]] for j in range(len_)] for i in inputs]
  else:
    len_ = len(inputs)
    index = list(range(len_))
    shuffle(index)
    outputs = [inputs[i] for i in index]
  return outputs


class DSBuilder(object):
  """
    Data Set Builder

    Argu:
      dsdir: Str. Path of Dataset.
  """

  def __init__(self, dsdir, size:list, shuffle=False, pklen=None, pklname='filelist.txt'):
    
    self.dsdir = dsdir
    self.size = size
    self.shuffle = shuffle
    self.pklen = pklen or self._cpklen(size)
    self.pklname = pklname

    self.classes_dict = {}

    self.get_classes_dict()
    
  def _cpklen(self, size):
    """
      Compute pkl lenth

      Max Byte is 2GiB(2^31).
    """
    maxbyte = 2 ** 31
    if len(size) == 2:
      photo = size[0] * size[1] * 3
    else:
      photo = size[0] * size[1] * size[2]
    return maxbyte // photo

  def get_classes_dict(self, filename='classes.txt'):
    """
      Get Classes Dict

      Argu:
        filename: Str. File name of the file that stores the 
        classified information.

      Return:
        Dict.
    """
    if not self.classes_dict:
      with open(f"{self.dsdir}/{filename}", 'r') as f:
        temp = [i.strip() for i in f.readlines()]
      self.classes_dict = dict(zip(range(len(temp)), temp))
    return self.classes_dict

  def get_all(self, mode):
    """
      mode:
        ignore
        fill0
        fillx
        stretch
        crop
    """
    train, val, test = self.load()
    if any([train, val, test]):
      train_x, train_y = train
      val_x, val_y = val
      test_x = test
    else:
      train_x, train_y = self.get_data('train', mode)
      val_x, val_y = self.get_data('val', mode)
      test_x = self.get_data('test', mode)
      self.save(
        [train_x, train_y],
        [val_x, val_y],
        test_x)
    return [train_x, train_y], [val_x, val_y], test_x

  def get_data(self, name, img_mode, suffix='.jpg'):
    if name != 'test':
      images, labels = [], []
      for item in self.classes_dict:
        _dir = f"{self.dsdir}/{name}/{self.classes_dict[item]}/"
        for i in range(len(os.listdir(_dir))):
          img = self.img_func(_dir + f'{i}{suffix}', img_mode)
          images.append(img)
          labels.append(item)
      if self.shuffle:
        images, labels = _shuffle([images, labels])
      return np.array(images), np.array(labels)
    else:
      images = []
      _dir = f"{self.dsdir}/{name}/"
      if not os.path.exists(_dir):
        return None
      for files in os.listdir(_dir):
          img = self.img_func(_dir + files, img_mode)
          images.append(img)
      return np.array(images)

  def save(self, train, val, test=None, suffix='.gz'):
    
    file_list = []
    
    num_train = len(train[0])
    for i in range((num_train + self.pklen - 1) // self.pklen):
      if (i + 1) * self.pklen <= num_train:
        dtrain = {
          'train_x': train[0][i*self.pklen:(i + 1)*self.pklen],
          'train_y': train[1][i*self.pklen:(i + 1)*self.pklen],
        }
      else:
        dtrain = {
          'train_x': train[0][i*self.pklen:],
          'train_y': train[1][i*self.pklen:],
        }
      with gzip.open(f"{self.dsdir}/train{i}{suffix}", 'wb') as f:
        pickle.dump(dtrain, f)
      file_list.append(f'train{i}{suffix}\n')

    num_val = len(val[0])
    for i in range((num_val + self.pklen - 1) // self.pklen):
      if (i + 1) * self.pklen <= num_val:
        dval = {
          'val_x': val[0][i*self.pklen:(i + 1)*self.pklen],
          'val_y': val[1][i*self.pklen:(i + 1)*self.pklen],
        }
      else:
        dval = {
          'val_x': val[0][i*self.pklen:],
          'val_y': val[1][i*self.pklen:],
        }
      with gzip.open(f"{self.dsdir}/val{i}{suffix}", 'wb') as f:
        pickle.dump(dval, f)
      file_list.append(f'val{i}{suffix}\n')

    if test != None:
      num_test = len(test)
      for i in range((num_test + self.pklen - 1) // self.pklen):
        if (i + 1) * self.pklen <= num_test:
          dtest = {
            'test_x': test[i*self.pklen:(i + 1)*self.pklen],
          }
        else:
          dtest = {
            'test_x': test[i*self.pklen:],
          }
        with gzip.open(f"{self.dsdir}/test{i}{suffix}", 'wb') as f:
          pickle.dump(dtest, f)
        file_list.append(f'test{i}{suffix}\n')
    
    with open(f"{self.dsdir}/{self.pklname}", 'w') as f:
      f.writelines(file_list)

  def load(self):
    if not os.path.exists(f"{self.dsdir}/{self.pklname}"):
      return [], [], []
    
    with open(f"{self.dsdir}/{self.pklname}", 'r') as f:
      filelist = [i.strip() for i in f.readlines()]

    train_list = [i for i in filelist if 'train' in i]
    train_x, train_y = [], []
    for i in train_list:
      with gzip.open(f"{self.dsdir}/{i}", 'rb') as f:
        train = pickle.load(f)
        train_x.append(train['train_x'])
        train_y.append(train['train_y'])
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)

    val_list = [i for i in filelist if 'val' in i]
    val_x, val_y = [], []
    for i in val_list:
      with gzip.open(f"{self.dsdir}/{i}", 'rb') as f:
        val = pickle.load(f)
        val_x.append(val['val_x'])
        val_y.append(val['val_y'])
    val_x = np.concatenate(val_x)
    val_y = np.concatenate(val_y)

    test_list = [i for i in filelist if 'test' in i]
    if test_list:
      test_x = []
      for i in test_list:
        with gzip.open(f"{self.dsdir}/{i}", 'rb') as f:
          test = pickle.load(f)
          test_x.append(test['test_x'])
      test_x = np.concatenate(test_x)
    else:
      test_x = None

    return [train_x, train_y], [val_x, val_y], test_x

  def img_func(self, filename, mode):
    img = Image.open(filename)

    if img.mode != 'RGB':
      print(filename, img.mode)
      img.convert('RGB')
    
    if mode == 'ignore':
      img = np.array(img)
    elif mode == 'fill0':
      w, h = img.size
      
      if h > self.size[0] or w > self.size[1]:
        if h >= w:
          _w = int(w / h * self.size[0])
          img = img.resize((_w, self.size[0]))
        if w > h:
          _h = int(h / w * self.size[1])
          img = img.resize((self.size[1], _h))

      w, h = img.size
      p_w = self.size[0] - w
      p_h = self.size[1] - h
      pad_w = (int(p_w / 2), p_w - int(p_w / 2))
      pad_h = (int(p_h / 2), p_h - int(p_h / 2))

      img = np.array(img)
      img = np.pad(img, (pad_h, pad_w, (0, 0)), 'constant', constant_values=0)
    elif mode == 'fillx':
      w, h = img.size
      
      if h > self.size[0] or w > self.size[1]:
        if h >= w:
          _w = int(w / h * self.size[0])
          img = img.resize((_w, self.size[0]))
        if w > h:
          _h = int(h / w * self.size[1])
          img = img.resize((self.size[1], _h))

      w, h = img.size
      p_w = self.size[0] - w
      p_h = self.size[1] - h
      pad_w = (int(p_w / 2), p_w - int(p_w / 2))
      pad_h = (int(p_h / 2), p_h - int(p_h / 2))

      img = np.array(img)
      img = np.pad(img, (pad_h, pad_w, (0, 0)), 'linear_ramp')
    elif mode == 'stretch':
      img = img.resize((self.size[0], self.size[1]))
      img = np.array(img)
    elif mode == 'crop':
      w, h = img.size
      img = np.array(img)
      if h > self.size[0]:
        ph = h - self.size[0]
        lh = [ph // 2, ph - ph // 2]
        img = img[lh[0]:h - lh[1],:]
      if w > self.size[1]:
        pw = w - self.size[1]
        lw = [pw // 2, pw - pw // 2]
        img = img[:,lw[0]:w - lw[1]]
      if h < self.size[0]:
        ph = self.size[0] - h
        lh = [ph // 2, ph - ph // 2]
        img = np.pad(img, (lh, 0, (0, 0)), 'constant', constant_values=0)
      if w < self.size[1]:
        pw = self.size[1] - w
        lw = [pw // 2, pw - pw // 2]
        img = np.pad(img, (0, lw, (0, 0)), 'constant', constant_values=0)

    return img


if __name__ == "__main__":
  from pprint import pprint
  
  foo = list(range(10))
  bar = list(range(10))
  bar.reverse()
  pprint(_shuffle(foo, False))
  pprint(_shuffle([foo, bar]))
  
  dsb = DSBuilder('datasets/car10x', [256, 256], True)
  pprint(dsb.get_classes_dict())
  print(dsb.pklen)
  
  p1 = dsb.img_func('datasets/car10x/val/bus/0.jpg', 'ignore')
  p2 = dsb.img_func('datasets/car10x/val/bus/0.jpg', 'fill0')
  p3 = dsb.img_func('datasets/car10x/val/bus/0.jpg', 'fillx')
  p4 = dsb.img_func('datasets/car10x/val/bus/0.jpg', 'stretch')
  print(p1.shape, p2.shape, p3.shape, p4.shape)
  Image.fromarray(p1).show()
  Image.fromarray(p2).show()
  Image.fromarray(p3).show()
  Image.fromarray(p4).show()

  print(dsb.get_data('val', 'fillx')[0].shape)
