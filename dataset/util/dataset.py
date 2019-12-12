# -*- coding: utf-8 -*-
"""Dataset

  File: 
    /hat/dataset/util/dataset

  Description: 
    Dataset 基类
"""


# import setting
__all__ = [
    'Dataset']


class Dataset(object):
  """
    Dataset

    数据集基类。需要重写self.args()方法。
  """
  def __init__(self, dtype=None, **kwargs):
    # set Config
    self.dtype = dtype or 'float32'
    # set dataset parameters
    self.mission_list = []
    self.num_train = 0
    self.num_val = 0
    self.num_test = 0
    self.input_shape = ()
    self.output_shape = ()
    self.train_x = None
    self.train_y = None
    self.val_x = None
    self.val_y = None
    self.test_x = None
    self.test_y = None

    # self.parameter_dict = {
    #   'mission_list': [],
    #   'num_train': 0,
    #   'num_val': 0,
    #   'num_test': 0,
    #   'input_shape': (),
    #   'output_shape': (),
    #   'train_x': None,
    #   'train_y': None,
    #   'val_x': None,
    #   'val_y': None,
    #   'test_x': None,
    #   'test_y': None,
    # }
    # default parameters
    # for item in self.parameter_dict:
    #   self.__dict__[item] = self.parameter_dict[item]
    # dataset parameters
    self.args()
    # transform dtype
    self.set_dtype()
    # feedback to config
    # for item in self.parameter_dict:
    #   self.config.__dict__[item] = self.__dict__[item]
      # self.config._set(item, self.__dict__[item])

  def set_dtype(self):
    for item in ['train_x', 'train_y', 'val_x', 'val_y', 'test_x', 'test_y']:
      if self.__dict__[item] is not None:
        self.__dict__[item] = self.__dict__[item].astype(self.dtype)

  def args(self):
    raise NotImplementedError


# test
if __name__ == "__main__":
  # from hat.utils.tconfig import tconfig
  # a = tconfig()
  # d = Dataset(a)
  # print(d.input_shape)
  # print(a.input_shape)
  pass
