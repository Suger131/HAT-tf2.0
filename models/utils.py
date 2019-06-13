import importlib


__all__ = ['MLib']


def MLib(lib=''):
  """
    Import Model Lib, you can use short name or full name.

    ```
    S: Standard (default)
    A: Alpha
    B: Beta
    ```
  """
  if lib == 'S':
    lib = 'standard'
  elif lib == 'A':
    lib = 'alpha'
  elif lib == 'B':
    lib = 'beta'
  try:
    return importlib.import_module(f'hat.models.{lib}')
  except:
    raise Exception(f'{lib} not in Model Lib')

if __name__ == "__main__":
  models = MLib('S')
  # mod = getattr(models, 'mlp')(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  # lenet = importlib.import_module(f'hat.models.lenet')
  # mod2 = lenet.lenet(DATAINFO={'INPUT_SHAPE': (32, 32, 3), 'NUM_CLASSES': 10})
  # print(dir(lenet))
  print(getattr(models, '213'))