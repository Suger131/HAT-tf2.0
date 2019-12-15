# -*- coding: utf-8 -*-
"""Config

  File: 
    /hat/util/config

  Description: 
    config tools
"""


# import setting
__all__ = [
    'Config',]


import os

import tensorflow as tf

from hat.util import __config__ as C
from hat.util import importer
from hat.util import log
from hat.util import time


class Config(object):
  """Config
  
    Description: 
      参数管理工具，包含的参数: 
        dataset
        model
        train param
        dirs
        gpu info

    Attributes:
      None

    Usage:
    ```python
      import hat
      c = hat.Config()
      c = hat.util.Config()  # or
      c = hat.util.config.Config()  # or
      # or
      from hat.util import config
      c = config.Config()
    ```
  """
  def __init__(self):
    self.raw_input = input('=>')
    # ========================
    # Default Parameters
    # ========================
    self.dataset_name = ''
    self.lib_name = ''
    self.model_name = ''
    self.batch_size = 0
    self.epochs = 0
    self.step = 0
    self.step_per_log = 0
    self.step_per_val = 0
    self.opt = ''
    self.loss = ''
    self.metrics = []
    self.dtype = ''
    self.aug = None
    self.is_train = False
    self.is_val = False
    self.is_test = False
    self.is_save = False
    self.is_save_best = False
    self.is_gimage = False
    self.is_flops = False
    self.is_enhance = False
    self.is_write_middle_data = False
    self.is_write_loss = False
    self.run_mode = ''
    self.addition = ''
    self.lr_alt = False
    self.__dict__ = {**self.__dict__, **C.get('default')}
    self._name_map = C.get('name_map')

    # ========================
    # Empty Parameters
    # ========================
    self._warning_list = []
    self._input_late_parameters = {}
    self._logc = []
    # gpu info
    self.gpus = tf.config.experimental.list_physical_devices('GPU')
    self.gpu_growth = C.get('gpu_growth')
    self.xgpu = False
    self.xgpu_num = 0
    self.xgpu_max = len(self.gpus)
    # objs
    self.dataset = None
    self.data = None
    self.model = None
    # dirs
    self.log_root = C.get('log_root')
    self.log_dir = ''
    self.h5_name = ''
    self.load_name = ''
    self.save_name = ''
    self.save_dir = ''
    self.save_time = 1
    self.tb_dir = ''
    # dataset parameters
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

    # ========================
    # Processing Parameters
    # ========================
    self.processing_input()
    self.processing_envs()

  def processing_input(self):
    in_list = self.raw_input.split(' ')
    for i in in_list:
      if i in ['', ' ']:
        continue
      if '=' in i:
        temp = i.split('=')
        if temp[0] not in self._name_map:
          self._warning_list.append(f'Unsupported option: {temp[0]}')
          continue
        var = self._name_map[temp[0]]
        if 'd' in var:
          self._warning_list.append(f"Can't assign a tag: {temp[0]}")
          continue
        data = temp[1]
        var_name = var['n']
        var_data = int(data) if type(data) == str and data.isdigit() and 'force_str' not in var else data
        if 'l' in var:
          self._input_late_parameters[var_name] = var_data
        else:
          self.__dict__[var_name] = var_data
        continue
      if i not in self._name_map:
        self._warning_list.append(f'Unsupported option: {i}')
        continue
      var = self._name_map[i]
      self.__dict__[var['n']] = var['d']

  def processing_envs(self):
    # lib
    self.lib_name = importer.get_fullname(self.lib_name)
    # xgpu
    self.set_xgpu()
    # dir
    self.set_dirs()
    # log
    self.set_logger()
    try:
      # dataset
      self.set_dataset()
      # model
      self.set_model()
    except Exception:
      # In case of unknown errors
      log.exception(name=__name__, exit=True)
    # log other param
    self.log_param()

  def set_gpu_memory_growth(self):
    if self.gpu_growth:
      for gpu in self.gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

  def set_xgpu(self):
    if self.xgpu:
      if isinstance(self.xgpu, int):
        if self.xgpu == -1:
          self.xgpu_num = self.xgpu_max
        elif self.xgpu > self.xgpu_max:
          self.xgpu_num = self.xgpu_max
          # warning
        else:
          self.xgpu_num = self.xgpu
        self.xgpu = True
        # TODO: XGPU
      else:
        pass  # warning

  def set_dirs(self):
    self.save_dir = os.path.join(
        self.log_root,
        self.lib_name,
        f"{self.model_name}_{self.dataset_name}")
    if self.addition:
      self.save_dir += '_' + self.addition
    self.log_dir = os.path.join(
        self.save_dir,
        C.get('log_name'))
    self.h5_name = os.path.join(
        self.save_dir,
        C.get('h5_name'))
    if not os.path.exists(self.save_dir):
      os.makedirs(self.save_dir)
    # TODO: change save name of weights
    self.save_time = len([item for item in os.listdir(self.save_dir)
        if C.get('h5_name') in item])  # 获取save dir下有多少h5文件
    self.save_name = f'{self.h5_name}_{self.save_time + 1}.h5'
    self.load_name = f'{self.h5_name}_{self.save_time}.h5'
    self.tb_dir = os.path.join(
        self.save_dir,
        C.get('tb_dir'))

  def set_logger(self):
    log.init(log_dir=self.log_dir)
    log.info(f'Logs dir: {self.log_dir}', name=__name__)
    log.log_list(self._warning_list, leval='warn', name=__name__)

  def set_dataset(self):
    log.info(f"Loading Dataset: {self.dataset_name}", name=__name__)
    dataset_caller = importer.load('d', self.dataset_name)
    self.dataset = dataset_caller(dtype=self.dtype)
    log.info(f"Loaded Dataset: {self.dataset_name}", name=__name__)
    self.data = self.dataset

  def set_model(self):
    log.info(f"Model Lib: {self.lib_name}", name=__name__)
    log.info(f"Loading Model: {self.model_name}", name=__name__)
    model_caller = importer.load(self.lib_name, self.model_name)
    model_caller(config=self)
    for item in self._input_late_parameters:
      if item == 'opt': continue
      self.__dict__[item] = self._input_late_parameters[item]
    self.model.compile(
      optimizer=self.opt,
      loss=self.loss,
      metrics=self.metrics,)
    log.info(f"Loaded Model: {self.model_name}", name=__name__)

  def log_param(self):
    if self.run_mode == 'no-gimage':
      self.is_gimage = False
      log.info('Do not get model image.', name=__name__)
    if self.run_mode == 'gimage':
      self.is_train = False
      self.is_val = False
      self.is_save = False
      log.info('Get model image only.', name=__name__)
    if self.run_mode == 'trian':
      self.is_val = False
      log.info('Train only.', name=__name__)
    if self.run_mode == 'val':
      self.is_train = False
      self.is_save = False
      log.info('Val only.', name=__name__)
    if self.run_mode != 'gimage':
      log.info(f"Batch size: {self.batch_size}", name=__name__)
      if not self.step:
        log.info(f"Epochs: {self.epochs}", name=__name__)
      else:
        log.info(f"Steps: {self.step}", name=__name__)
    if self.is_enhance:
      log.info('Enhance data.', name=__name__)
    if self.xgpu:
      log.info(f'XGPU On. GPU: {self.xgpu_num}', name=__name__)
    if self.lr_alt:
      log.info('Learning Rate Alterable.', name=__name__)


# test part
if __name__ == "__main__":
  c = Config()
  c.model.summary()
  # print({item: c.__dict__[item] for item in c.__dict__ if item != '_name_map'})

