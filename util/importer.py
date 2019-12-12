# -*- coding: utf-8 -*-
"""Importer

  File: 
    /hat/util/importer

  Description: 
    import tools
    Import工具，可加载自定义的dataset和model
    加载的方式是猜想和枚举，优先根据名字猜想文件，找不到才会
    通过枚举的方式搜索
    dataset路径: `hat.dataset.lib`
    model路径: `hat.model.{lib name}`
  
  Maps:
    dataset:
      D/d: dataset
    model:
      S/s: standard
      A/a: alpha
      B/b: beta
      T/t: test
"""


# import setting
__all__ = [
    'get_fullname',
    'get_lib_dir',
    'get_imp_like',
    'get_class',
    'load',]


import os
import importlib

from hat import __config__ as C
from hat.util import util
from hat.util import log


LIB_MAP = {
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
HAT_DIR = C.__root__
IGNORE = [
    '__init__.py',
    '__config__.py',
    '__pycache__']
DATASET_TUPLE = (
    'hat',
    'dataset',
    'lib')
MODEL_TUPLE = (
    'hat',
    'model')


def get_fullname(lib) -> str:
  """get_fullname"""
  return lib in LIB_MAP and LIB_MAP[lib] or lib


def get_lib_dir(lib) -> str:
  """get_lib_dir"""
  if lib == 'dataset':
    lib = os.path.join(
        HAT_DIR,
        *DATASET_TUPLE[1:])
  else:
    lib = os.path.join(
        HAT_DIR,
        MODEL_TUPLE[-1],
        lib)
  return lib


def get_imp_like(lib, name) -> str:
  """get_imp_like"""
  return '.'.join(lib == 'dataset' and DATASET_TUPLE + (name,)\
    or MODEL_TUPLE + (lib, name))


def get_class(lib, name):
  """get_class"""
  name = str(name)
  target = None
  try_bool = False
  for_name = ''
  
  # 猜想
  try_name = util.del_tail_digit(name)
  imp_name = get_imp_like(lib, try_name)
  spec = importlib.util.find_spec(imp_name)
  if spec:
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if name in dir(module):
      target = getattr(module, name)
    else:
      try_bool = True
  if not target:  # 遍历
    filelist = os.listdir(get_lib_dir(lib))
    for key in IGNORE:
      if key in filelist:
        filelist.remove(key)
    for item in filelist:
      item_name = get_imp_like(lib, item.strip('.py'))
      spec = importlib.util.find_spec(item_name)
      if spec:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if name in dir(module):
          target = getattr(module, name)
          for_name = item.strip('.py')
          break
 
  if target is None:
    if lib == 'dataset':
      log.error(f"[ImportError] '{name}' not in dataset",
          exit=True, name=__name__)
    else:
      log.error(f"[ImportError] '{name}' not in model.{lib}",
          exit=True, name=__name__)
  elif try_bool:
    log.warn(f"{name} in '{for_name}' but not '{try_name}'. " \
        f"Please comply with the naming standard.", name=__name__)
  return target


def load(lib='', name=''):
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
  lib = get_fullname(lib)
  return get_class(lib, name)


# test
if __name__ == "__main__":
  from hat.util import test
  log.init('./unpush/test')
  tc = test.TestConfig()
  print(get_imp_like('dataset', 'mnist'))
  print(get_imp_like('standard', 'mlp'))
  print(get_lib_dir('standard'))
  target1 = load('d', 'mnist')
  target1(tc)
  print(tc.train_x.shape)
  target2 = load('s', 'lenet')
  target2(config=tc)
  # tc.model.summary()
  target3 = load('s', 'untitled')
  target4 = load('d', 'untitled')
  # data = i.load('d', 'mnist')(tc)
  # print(tc.train_x.shape)
  # model = i.load('s', 'mlp')(tc)
  # tc.model.summary()

