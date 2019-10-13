"""
  hat.model.utils.importer
"""

# pylint: disable=unused-argument


__all__ = [
  'importer',
]


import importlib


class importer(object):
  """
    importer
  """
  def __init__(self, *args, **kwargs):
    self.lib_dict = {
      'S': 'standard',
      's': 'standard',
      'A': 'alpha',
      'a': 'alpha',
      'B': 'beta',
      'b': 'beta',
      'T': 'test',
      't': 'test',
    }

  def mlib(self, lib='s'):
    """
      Import Model lib, you can use short name or full name.

      ```
        S/s: standard #default
        A/a: alpha
        B/b: beta
        T/t: test
      ```
    """
    if lib in self.lib_dict:
      lib = self.lib_dict[lib]
    try:
      return importlib.import_module(f'hat.model.{lib}')
    except:
      raise Exception(f'{lib} not in Model lib')

  def nlib(self, lib):
    if lib in self.lib_dict:
      lib = self.lib_dict[lib]
    return lib


# test
if __name__ == "__main__":
  lib_name = 's'
  print(importer().nlib(lib_name))
