import importlib


__all__ = ['MLib', 'NLib']


LIB_DICT={
  'S': 'standard',
  'A': 'alpha',
  'B': 'beta',
  'T': 'test',
}


def MLib(lib=''):
  """
    Import Model Lib, you can use short name or full name.

    ```
    S: Standard (default)
    A: Alpha
    B: Beta
    T: Test
    ```
  """
  if lib in LIB_DICT:
    lib = LIB_DICT[lib]
  try:
    return importlib.import_module(f'hat.models.{lib}')
  except:
    raise Exception(f'{lib} not in Model Lib')


def NLib(lib):
  if lib in LIB_DICT:
    lib = LIB_DICT[lib]
  return lib


if __name__ == "__main__":
  models = MLib('S')
  # mod = getattr(models, 'mlp')(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  # lenet = importlib.import_module(f'hat.models.lenet')
  # mod2 = lenet.lenet(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  # print(dir(lenet))
  # print(getattr(models, '213'))
  print(NLib('S'))
  print('S' in LIB_DICT, 'C' in LIB_DICT)
