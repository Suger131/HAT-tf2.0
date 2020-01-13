# -*- coding: utf-8 -*-
"""Builder

  File: 
    /hat/dataset/builder

  Description: 
    自定义数据集生成/读取器
"""


# import setting
__all__ = [
    'DatasetBuilder',]


import gzip
import os
import pickle

import numpy as np
from PIL import Image

from hat import __config__ as C
from hat.dataset import util

# def _shuffle(inputs, islist=True):
#   """_shuffle

#     Description:
#       Shuffle the datas/labels
#       Private method

#     Args:
#       inputs: np.array/list, or list/tuple of np.array/list. 
#       In the latter case, the corresponding elements for each 
#       item in the list are shuffled together as a whole.
#       islist: Boolean. The latter case is enabled when the 
#       value is True. Default is True.

#     Return:
#       A shuffled np.array/list(depend on argus)
#   """
#   from random import shuffle
#   if islist:
#     len_ = len(inputs[0])
#     index = list(range(len_))
#     shuffle(index)
#     outputs = [[i[index[j]] for j in range(len_)] for i in inputs]
#   else:
#     len_ = len(inputs)
#     index = list(range(len_))
#     shuffle(index)
#     outputs = [inputs[i] for i in index]
#   return outputs


class DatasetBuilder(object):
  """
    Data Set Builder

    Args:
      dsdir: Str. Path of Dataset.
  """
  def __init__(self,
    dsdir: str,
    size: list,
    shuffle=False,
    pklen=None,
    pklname='filelist',
    suffix='.txt',
    **kwargs):
    self.dsdir = dsdir
    self.size = size
    self.shuffle = shuffle
    self.pklen = pklen or self._cpklen(size)
    self.pklname = pklname + suffix
    self.suffix = suffix

    self.classes_dict = {}
    self.get_classes_dict()

  # Built-in method

  def _cpklen(self, size):
    """
      Compute pkl lenth

      Max Byte is 2GiB(2^31).
    """
    maxbyte = C.get('maxbyte')
    if len(size) == 2:
      photo = size[0] * size[1] * 3
    else:
      photo = size[0] * size[1] * size[2]
    return maxbyte // photo

  def _img_func(self, filename, mode):
    
    ## load image
    img = Image.open(filename)
    
    ## transform image mode if necessary
    if img.mode != 'RGB':
      print(filename, img.mode)
      img = img.convert('RGB')
    
    ## mode: ignore
    if mode in ['ignore', 'ig']:
      img = np.array(img)
      # output
      return img

    ## mode: fill0(f0)
    if mode in ['fill0', 'f0']:
      # resize
      w, h = img.size
      if h > self.size[0] or w > self.size[1]:
        if h >= w:
          _w = int(w / h * self.size[0])
          img = img.resize((_w, self.size[0]))
        if w > h:
          _h = int(h / w * self.size[1])
          img = img.resize((self.size[1], _h))
      # fill with 0
      w, h = img.size
      p_w = self.size[0] - w
      p_h = self.size[1] - h
      pad_w = (int(p_w / 2), p_w - int(p_w / 2))
      pad_h = (int(p_h / 2), p_h - int(p_h / 2))
      img = np.array(img)
      img = np.pad(img, (pad_h, pad_w, (0, 0)), 'constant', constant_values=0)
      # output
      return img

    ## mode: fillx(fx)
    if mode in ['fillx', 'fx']:
      # resize
      w, h = img.size
      if h > self.size[0] or w > self.size[1]:
        if h >= w:
          _w = int(w / h * self.size[0])
          img = img.resize((_w, self.size[0]))
        if w > h:
          _h = int(h / w * self.size[1])
          img = img.resize((self.size[1], _h))
      # fillx
      w, h = img.size
      p_w = self.size[0] - w
      p_h = self.size[1] - h
      pad_w = (int(p_w / 2), p_w - int(p_w / 2))
      pad_h = (int(p_h / 2), p_h - int(p_h / 2))
      img = np.array(img)
      img = np.pad(img, (pad_h, pad_w, (0, 0)), 'linear_ramp')
      # output
      return img

    ## mode: stretch
    if mode in ['stretch', 's']:
      img = img.resize((self.size[0], self.size[1]))
      img = np.array(img)
      # output
      return img

    ## mode: crop
    if mode in ['crop', 'c']:
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
      # output
      return img

    ## mode: imagenet
    if mode in ['imagenet', 'in']:
      w, h = img.size
      if w <= h:
        nw = 256
        nh = int(h / w * 256)
      else:
        nw = int(w / h * 256)
        nh = 256
      img = img.resize((nw, nh))
      img = np.array(img)
      ph = abs(nh - self.size[0])
      lh = [ph // 2, ph - ph // 2]
      pw = abs(nw - self.size[1])
      lw = [pw // 2, pw - pw // 2]
      if nh >= self.size[0]:
        img = img[lh[0]:nh - lh[1],:]
      else:
        img = np.pad(img, (lh, 0, (0, 0)), 'constant', constant_values=0)
      if nw >= self.size[1]:
        img = img[:,lw[0]:nw - lw[1]]
      else:
        img = np.pad(img, (0, lw, (0, 0)), 'constant', constant_values=0)
      return img

    raise Exception('An incorrect mode was passed in:', mode)

  def _load(self):
    if not os.path.exists(f"{self.dsdir}/{self.pklname}"):
      return [], [], []
    
    with open(f"{self.dsdir}/{self.pklname}", 'r') as f:
      filelist = [i.strip() for i in f.readlines()]

    # NOTE: NTF
    # if num of *.gz more than 2
    # turn to datagenerator

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

  def _save(self, data, suffix='.gz'):

    train, val, test = data
    file_list = []

    ## train
    num_train = len(train[0])
    for i in range((num_train + self.pklen - 1) // self.pklen):
      if (i + 1) * self.pklen <= num_train:
        dtrain = {
            'train_x': train[0][i*self.pklen:(i + 1)*self.pklen],
            'train_y': train[1][i*self.pklen:(i + 1)*self.pklen],}
      else:
        dtrain = {
            'train_x': train[0][i*self.pklen:],
            'train_y': train[1][i*self.pklen:],}
      with gzip.open(f"{self.dsdir}/train{i}{suffix}", 'wb') as f:
        pickle.dump(dtrain, f)
      file_list.append(f'train{i}{suffix}\n')

    ## val
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

    ## test
    if test != None:
      num_test = len(test)
      for i in range((num_test + self.pklen - 1) // self.pklen):
        if (i + 1) * self.pklen <= num_test:
          dtest = {
              'test_x': test[i*self.pklen:(i + 1)*self.pklen],}
        else:
          dtest = {
              'test_x': test[i*self.pklen:],}
        with gzip.open(f"{self.dsdir}/test{i}{suffix}", 'wb') as f:
          pickle.dump(dtest, f)
        file_list.append(f'test{i}{suffix}\n')
    
    ## write pklfile
    with open(f"{self.dsdir}/{self.pklname}", 'w') as f:
      f.writelines(file_list)

  def _get_data(self, name, img_mode, suffix='.jpg'):

    # NOTE: NTF
    # 根据imagenet的todo
    # 将标准化也整合到这里
    # 对于整理程度比较高的数据集可以直接自动化建库
    # 对于整理程度不够高的数据集可以先预处理再自动化建库

    if name != 'test':
      images, labels = [], []
      for item in self.classes_dict:
        _dir = f"{self.dsdir}/{name}/{self.classes_dict[item]}/"
        for i in os.listdir(_dir):
          img = self._img_func(_dir + i + suffix, img_mode)
          images.append(img)
          labels.append(item)
      if self.shuffle:
        images, labels = util.shuffle([images, labels])
      return np.array(images), np.array(labels)
    else:
      images = []
      _dir = f"{self.dsdir}/{name}/"
      if not os.path.exists(_dir):
        return None
      for files in os.listdir(_dir):
        img = self._img_func(_dir + files, img_mode)
        images.append(img)
      return np.array(images)

  # Public method

  def get_classes_dict(self, filename='classes'):
    """
      Get Classes Dict

      Argu:
        filename: Str. File name of the file that stores the 
        classified information.

      Return:
        Dict.
    """
    filename = filename + self.suffix
    if not self.classes_dict:
      with open(f"{self.dsdir}/{filename}", 'r') as f:
        temp = [i.strip() for i in f.readlines()]
      self.classes_dict = dict(zip(range(len(temp)), temp))
    return self.classes_dict

  def get(self, mode):
    """
      Args:
        mode: Str. Include: ignore/ig, fill0/f0, fillx/fx, stretch/s, crop/c, imagenet/in

      Return:
        Tuple: ([train_x, train_y], [val_x, val_y], test_x) # Numpy.array
    """
    train, val, test = self._load()
    if any([train, val, test]):
      train_x, train_y = train
      val_x, val_y = val
      test_x = test
    else:
      train_x, train_y = self._get_data('train', mode)
      val_x, val_y = self._get_data('val', mode)
      test_x = self._get_data('test', mode)
      self._save([
          [train_x, train_y],
          [val_x, val_y],
          test_x])
    return [train_x, train_y], [val_x, val_y], test_x


