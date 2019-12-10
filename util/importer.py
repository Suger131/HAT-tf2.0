# -*- coding: utf-8 -*-
"""Importer

  File: 
    /hat/util/importer

  Description: 
    import tools
"""


# import setting
__all__ = [
  'Importer',
]


import os
import importlib

from hat.util import util


class Importer(object):
  """Importer

    Description: 
      Import工具，可加载自定义的dataset和model
      加载的方式是猜想和枚举，优先根据名字猜想文件，找不到才会
      通过枚举的方式搜索
      dataset路径: `hat.dataset.lib`
      model路径: `hat.model.{lib name}`
      
    Attributes:
      config: hat.Config. 

    Maps:
      dataset:
        D/d: dataset
      model:
        S/s: standard
        A/a: alpha
        B/b: beta
        T/t: test
  """
  def __init__(self, config, *args, **kwargs):
    self.config = config
    self.lib_maps = {
      'D': 'dataset',
      'd': 'dataset',
      'S': 'standard',
      's': 'standard',
      'A': 'alpha',
      'a': 'alpha',
      'B': 'beta',
      'b': 'beta',
      'T': 'test',
      't': 'test',}
    # NOTE: this file dir is 'hat\util\importer.py'
    self.hat_dir = os.path.dirname(os.path.dirname(\
        os.path.abspath(__file__)))
    self.ignore = ['__init__.py', '__pycache__']

  def load(self, lib='', name=''):
    """Import target class from lib. 

      Description: 
        lib可以使用全名或缩写

      Args:
        lib: Str. 库名
        name: Str. 需要加载的dataset/model的名字

      Return:
        hat.Dataset or hat.Network or None

      Raises:
        ImportError
    """
    lib = self.get_fullname(lib)
    return self.get_class(lib, name)

  def get_fullname(self, lib):
    if lib in self.lib_maps:
      lib = self.lib_maps[lib]
    return lib

  def get_lib_dir(self, lib):
    if lib == 'dataset':
      lib = f'{self.hat_dir}/dataset/lib'
    else:
      lib = f'{self.hat_dir}/model/{lib}'
    return lib

  def get_imp_like(self, lib, name):
    if lib == 'dataset':
      output = f'hat.dataset.lib.{name}'
    else:
      output = f'hat.model.{lib}.{name}'
    return output

  def get_class(self, lib, name):
    name = str(name)
    target = None
    try_bool = False
    for_name = ''
    # 猜想
    try:
      try_name = util.del_tail_digit(name)
      temp = importlib.import_module(self.get_imp_like(lib, try_name))
      try:
        target = getattr(temp, name)
      except:
        try_bool = True
    except:
      # 枚举
      filelist = os.listdir(self.get_lib_dir(lib))
      for key in self.ignore:
        filelist.remove(key)
      for item in filelist:
        try:
          temp = importlib.import_module(self.get_imp_like(lib, item.strip('.py')))
          try:
            target = getattr(temp, name)
            for_name = item
          except:
            pass
        except:
          pass
        if target is not None:
          break
    if target is None:
      self.config.log.error(f'[ImportError] {name} not in {lib} lib')
    if try_bool:
      self.config.log(f'{name} was not in {try_name} but in {for_name}.', a='Warning')
      self.config.log(f'Please check the naming specification.', a='Warning')
    return target


# test
if __name__ == "__main__":
  from hat.util import test
  tc = test.TestConfig()
  i = Importer(tc)
  data = i.load('d', 'mnist')(tc)
  print(tc.train_x.shape)
  model = i.load('s', 'mlp')(tc)
  tc.model.summary()

