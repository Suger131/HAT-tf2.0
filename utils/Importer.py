"""
  hat.utils.Importer
"""

# pylint: disable=unused-argument


__all__ = [
  'Importer',
]


import os
import importlib


def _del_digit(inputs:str):
  inx = 0
  for i in inputs:
    if i.isdigit():
      break
    inx += 1
  if inx != len(inputs):
    inputs = inputs[:inx]
  return inputs


class Importer(object):
  """
    Importer

    # dataset:
      import the dataset class from 'hat.dataset.lib'.

    # model:
      import the model class from the given lib. For example: load mlp from standard lib.
  """
  def __init__(self, config, *args, **kwargs):
    self.config = config
    self.lib_dict = {
      'D': 'dataset',
      'd': 'dataset',
      'S': 'standard',
      's': 'standard',
      'A': 'alpha',
      'a': 'alpha',
      'B': 'beta',
      'b': 'beta',
      'T': 'test',
      't': 'test',
    }
    # NOTE: this file dir is 'hat\utils\Importer.py'
    self._hat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    self._ignore = ['__init__.py', '__pycache__']

  def load(self, lib='', name=''):
    """
      Import target class from lib. 
      You can use short name or full name (lib).

      ```
        # dataset lib name
        D/d: dataset
        # model lib name
        S/s: standard
        A/a: alpha
        B/b: beta
        T/t: test
      ```
    """
    lib = self.get_fullname(lib)
    return self.get_class(lib, name)

  def get_fullname(self, lib):
    if lib in self.lib_dict:
      lib = self.lib_dict[lib]
    return lib

  def get_lib_dir(self, lib):
    if lib == 'dataset':
      lib = f'{self._hat_dir}\dataset\lib'
    else:
      lib = f'{self._hat_dir}\model\{lib}'
    return lib

  def get_imp_like(self, lib, name):
    if lib == 'dataset':
      output = f'hat.dataset.lib.{name}'
    else:
      output = f'hat.model.{lib}.{name}'
    return output

  def get_class(self, lib, name):
    target = None
    try_bool = False
    for_name = ''
    # 猜想
    try:
      try_name = _del_digit(name)
      temp = importlib.import_module(self.get_imp_like(lib, try_name))
      try:
        target = getattr(temp, name)
      except:
        try_bool = True
    except:
      # 枚举
      filelist = os.listdir(self.get_lib_dir(lib))
      for key in self._ignore:
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
      pass
    if target is None:
      self.config.log.error(f'{name} not in {lib} lib')
    if try_bool:
      self.config.log(f'{name} was not in {try_name} but in {for_name}.', a='Warning')
      self.config.log(f'Please check the naming specification.', a='Warning')
    return target


# test
if __name__ == "__main__":
  # lib_name = 's'
  # print(Importer().nlib(lib_name))
  # print('vgg16_32m'.strip('1234567890_'))
  # print(_del_digit('vgg16_32m'))

  ## this file dir is 'hat\utils\Importer.py'
  # _hat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  # ignore = ['__init__.py', '__pycache__']
  # filelist = os.listdir(f'{_hat_dir}\dataset\lib')
  # for key in ignore:
  #   filelist.remove(key)
  # print(filelist)
  from hat.utils.tconfig import tconfig
  tc = tconfig()
  i = Importer(tc)
  data = i.load('d', 'mnist')(tc)
  print(tc.train_x.shape)
  model = i.load('s', 'mlp')(tc)
  tc.model.summary()

