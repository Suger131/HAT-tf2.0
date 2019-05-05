import os
import configparser

from utils._func import _writer


class Config(object):

  def __init__(self, filename, ext='.ini', *args, **kwargs):
    self.config = configparser.ConfigParser()
    self.file = f'{filename}{ext}'
    self._create()
    self.config.read(self.file)
    self._log_num = len(self.config.sections())

  # private method

  def _check_section(self, name):
    if name not in self.config.sections():
      self.config.add_section(name)

  def _create(self):
    if not os.path.exists(self.file):
      with open(self.file, 'w') as f:
        f.write('')
    self._check_section('param')

  def _upgrade(self):
    with open(self.file, 'w') as f:
      self.config.write(f)

  # public method

  def param(self, dicts):
    self._check_section('param')
    for item in dicts:
      self.config.set('param', item, str(dicts[item]))
    self._upgrade()

  def log(self, dicts):
    self._check_section(f'log_{self._log_num}')
    for item in dicts:
      self.config.set(f'log_{self._log_num}', item, str(dicts[item]))
    self._upgrade()

  def get(self, section, name):
    _log_num = self._log_num
    if self.config.has_option(section, name):
      pass
    if section == 'param':
      pass
    else:
      if f'log_{self._log_num}' in self.config.sections():
        section = f'log_{self._log_num}'
      elif f'log_{self._log_num-1}' in self.config.sections():
        section = f'log_{self._log_num-1}'
      else:
        return None
    if name not in self.config.options(section):
      return None
    return self.config.get(section, name)

  def get_dict(self, section):
    if self.config.has_section(section):
      return {}
    return
    # to do

  def if_param(self):
    return self.config.options('param') and True or False


if __name__ == "__main__":
  a = Config('utils/test')
  a.param({'batch_size': 32, 'epochs': 16})
  print(a.get('param', 'epochs'))
  print(a.get('log,', 'x'))
  print(a.get('param', 'x'))
  a.log({'train_time': 15, 'global_steps': 17000})
