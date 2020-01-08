# pylint: disable=attribute-defined-outside-init
# pylint: disable=no-name-in-module

import gzip
import os
import pickle

import numpy as np
from PIL import Image
from random import shuffle

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from hat.datasets.Dataset import Dataset
from hat.datasets.utils import DG


class imagenet(Dataset):
  """
    ImageNet 数据集
  """

  def args(self):

    self.classes_dict = {}
    self.train_num = []
    self.train_filename = []
    self.train_list = []

    self._MISSION_LIST = ['classfication']
    self.SHUFFLE = True
    self.NUM_TRAIN = 1281167
    self.NUM_VAL = 50000
    self.NUM_TEST = 0
    self.NUM_CLASSES = 1000
    self.INPUT_SHAPE = (224, 224, 3)
    self.DATA_DIR = 'E:/1-ML/ImageNet'
    self.DATA_PKL_FILE = 'filelist.txt'
    self.TRAIN_PKL_NUM = 260
    self.VAL_PKL_NUM = 10

    self.trian_x = None
    self.train_y = None
    self.val_x = None
    self.val_y = None

    self.get_classes_dict()
    # self.load()

  def get_classes_dict(self, filename='ILSVRC2012_mapping.txt'):
    """
      Get Classes Dict

      Argu:
        filename: Str. File name of the file that stores the 
        classified information.

      Return:
        Dict.
    """
    with open(f"{self.DATA_DIR}/{filename}", 'r') as f:
      temp = [i.strip().split(' ')[-1] for i in f.readlines()]
    self.classes_dict = dict(zip(temp, range(len(temp))))
    return self.classes_dict

  def get_train_file(self):
    train_path = os.path.join(self.DATA_DIR, 'train')
    classes_list = os.listdir(train_path)
    train_num = []
    train_filename = []
    for classes_name in classes_list:
      dir_train_name = os.path.join(train_path, classes_name)
      _num = 0
      name_list = []
      for filename in os.listdir(dir_train_name):
        _num += 1
        name_list.append(f'{classes_name}/{filename}')
      shuffle(name_list)
      train_num.append(_num)
      train_filename.append(name_list)

    self.train_num = train_num
    self.train_filename = train_filename

    return self.train_num, self.train_filename

  def get_val_file(self):
    val_path = os.path.join(self.DATA_DIR, 'val')
    classes_list = os.listdir(val_path)
    val_filename = []
    for classes_name in classes_list:
      dir_val_name = os.path.join(val_path, classes_name)
      name_list = []
      for filename in os.listdir(dir_val_name):
        name_list.append(f'{classes_name}/{filename}')
      shuffle(name_list)
      val_filename.append(name_list)

    self.val_filename = val_filename

    return self.val_filename

  def gen_train_list(self):
    trian_list = []
    for i in self.train_num:
      alpha = i // self.TRAIN_PKL_NUM
      beta = i % self.TRAIN_PKL_NUM
      delta = [alpha + 1] * beta + [alpha] * (self.TRAIN_PKL_NUM - beta)
      shuffle(delta)
      trian_list.append(delta)
    self.train_list = trian_list
    return self.train_list

  def gen_packs(self):
    
    train_packs = []
    train_filename = [] + self.train_filename
    train_list_zip = list(zip(*self.train_list))
    for _l in train_list_zip:
      pack = []
      for inx, j in enumerate(_l):
        for k in range(j):
          pack.append(train_filename[inx].pop())
      shuffle(pack)
      train_packs.append(pack)
    self.train_packs = train_packs
    
    val_packs = []
    val_filename = [] + self.val_filename
    for i in range(self.VAL_PKL_NUM):
      pack = []
      for j in range(1000):
        for k in range(50 // self.VAL_PKL_NUM):
          pack.append(val_filename[j].pop())
      shuffle(pack)
      val_packs.append(pack)
    self.val_packs = val_packs

    dpacks = {
        'train_packs': self.train_packs,
        'val_packs': self.val_packs,
    }

    with gzip.open(os.path.join(self.DATA_DIR, 'packs.gz'), 'wb') as f:
        pickle.dump(dpacks, f)

    return self.train_packs, self.val_packs

  def gen_pkl(self, suffix='.gz'):

    filelist = []

    if not os.path.exists(os.path.join(self.DATA_DIR, 'pkl')):
      os.mkdir(os.path.join(self.DATA_DIR, 'pkl'))

    # val

    for inx, i in enumerate(self.val_packs):
      pkl_filename = os.path.join(self.DATA_DIR, 'pkl', f'val{inx}{suffix}')
      filelist.append(pkl_filename)
      if os.path.exists(pkl_filename):
        print(f'Skip: val pkl {inx}')
        continue
      images, labels = [], []
      for filename in i:
        img = self.img_func(f'{self.DATA_DIR}/val/{filename}')
        images.append(img)
        labels.append(self.classes_dict[filename.split('/')[0]])
      dval = {
        'val_x': np.array(images),
        'val_y': np.array(labels),
      }
      with gzip.open(pkl_filename, 'wb') as f:
        pickle.dump(dval, f)
      del dval
      print(f'Done: val pkl {inx}')

    # train

    for inx, i in enumerate(self.train_packs):

      # 分包操作，选择区间
      # if inx < 50:
      #   continue

      pkl_filename = os.path.join(self.DATA_DIR, 'pkl', f'train{inx}{suffix}')
      filelist.append(pkl_filename)
      if os.path.exists(pkl_filename):
        print(f'Skip: train pkl {inx}')
        continue
      images, labels = [], []
      for filename in i:
        img = self.img_func(f'{self.DATA_DIR}/train/{filename}')
        images.append(img)
        lab = self.classes_dict[filename.split('/')[0]]
        labels.append(lab)
      dtrain = {
        'train_x': np.array(images),
        'train_y': np.array(labels),
      }
      with gzip.open(pkl_filename, 'wb') as f:
        pickle.dump(dtrain, f)
      del dtrain
      print(f'Done: train pkl {inx}')
    
    with open(os.path.join(self.DATA_DIR, self.DATA_PKL_FILE), 'w') as f:
      f.writelines(filelist)

    return None

  def img_func(self, filename):
    """
      Reduce the image equidistant to a minimum edge of 256.

      And then crop out a 224x224 pixel picture in the center of the image.
    """
    img = Image.open(filename)

    if img.mode != 'RGB':
      if img.mode != 'L':
        print(filename, img.mode)
      img = img.convert('RGB')
      
    w, h = img.size
    if w <= h:
      nw = 256
      nh = int(h / w * 256)
    else:
      nw = int(w / h * 256)
      nh = 256
    img = img.resize((nw, nh))

    # w, h = img.size
    img = np.array(img)
    
    ph = abs(nh - self.INPUT_SHAPE[0])
    lh = [ph // 2, ph - ph // 2]
    pw = abs(nw - self.INPUT_SHAPE[1])
    lw = [pw // 2, pw - pw // 2]

    if nh >= self.INPUT_SHAPE[0]:
      img = img[lh[0]:nh - lh[1],:]
    else:
      img = np.pad(img, (lh, 0, (0, 0)), 'constant', constant_values=0)
    
    if nw >= self.INPUT_SHAPE[1]:
      img = img[:,lw[0]:nw - lw[1]]
    else:
      img = np.pad(img, (0, lw, (0, 0)), 'constant', constant_values=0)

    return img

  def data_generator(self, mode: str, batch_size: int, aug: ImageDataGenerator=None, 
          suffix='.gz'):
    """
      Data Generator

      Argu:
        mode: Str. 'train' or 'val'
        batch_size: Int.
        aug: ImageDataGenerator[tensorflow.python.keras.preprocessing.image.ImageDataGenerator]. 
             Whether to use Data Argument.
        suffix: Str.
    """
    data_len = {'train': self.NUM_TRAIN, 'val': self.NUM_VAL}[mode]
    return DG(os.path.join(self.DATA_DIR, 'pkl'), mode, batch_size, data_len, aug, suffix)

  def load(self):

    if not os.path.exists(os.path.join(self.DATA_DIR, self.DATA_PKL_FILE)):
      print("[DATASETS] Couldn't found PKL.")
      print("[DATASETS] Generate PKL.")
      packs_filename = os.path.join(self.DATA_DIR, 'packs.gz')
      if os.path.exists(packs_filename):
        with gzip.open(packs_filename, 'rb') as f:
          dpacks = pickle.load(f)
          self.train_packs = dpacks['train_packs']
          self.val_packs = dpacks['val_packs']
      else:
        self.get_train_file()
        self.get_val_file()
        self.gen_train_list()
        self.gen_packs()
      self.gen_pkl()

    return True

  def get_generator(self, batch_size: int, aug=None):
    self.trian_generator = self.data_generator('train', batch_size, aug)
    self.val_generator = self.data_generator('val', batch_size)
    return self.trian_generator, self.val_generator


# test mode
if __name__ == "__main__":
  m = imagenet()
  Gval = m.data_generator('val', 1024)
  for img, lab in Gval:
    print(len(img), len(lab))


