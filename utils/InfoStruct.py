"""
  hat.utils.InfoStruct
"""


def _getDecX(x:int, rounder=2):
  _Xlen = (len(str(int(x))) - 1) // 3
  _X = 'KMGTPEZY'[_Xlen - 1]
  _num = round(x / (10 ** (_Xlen * 3)), rounder)
  return f'{_num}{_X}'


def _getEX(x:float):
  return '%.1e' % x


class _InfoStruct(object):
  """
    Info Struct
  """
  def __init__(self, **kwargs):
    self._main_struct = []
    pass

  def mix(self, *args):
    """
      Mix Blocks

      # Parm
        *args: dict.

      # Return
        Tuple
    """
    return tuple(args)

  def link(self, key, value):
    """
      Link key value pair

      # Parm
        key: str.
        value: any.

      # Return
        dict
    """
    return {key:value}

  def gen(self, struct, tab=0):
    """
      Genetare the text of InfoStruct

      # Parm
        struct:tuple

      # Return
        Str.
    """
    temp = []
    if isinstance(struct, tuple) and tab == 0:
      for item in struct:
        temp.append(self.gen(item, tab))
    elif isinstance(struct, list):
      temp.append('\n' + '  ' * tab + '[\n')
      for item in struct:
        temp.append(self.gen(item, tab + 1))
      temp.append('  ' * tab + ']')
    elif isinstance(struct, dict):
      for item in struct:
        temp.append('  ' * tab)
        temp.append(f'{item}:')
        temp.append(self.gen(struct[item], tab))
        temp.append('\n')
    else:
      temp.append(str(struct))
    
    return ''.join(temp)

  def print(self, struct:tuple):
    """
      Print the InfoStruct

      # Parm
        struct:tuple
    """
    print(self.gen(struct))

  def config(self, inputs):
    """
      Config Block

      # Parm
        inputs: dict. Contains all key value pairs.

      # Return
        dict
    """

    _struct = []
    _autolist = [
      'dataset',
      'lib',
      'model',
      'xgpu',
      'batchsize',
      'loss',
      'opt',
      'input_shape',
    ]

    for item in _autolist:
      _temp = inputs.get(item)
      if _temp is None or _temp == False:
        _temp = 'no'
      if _temp == True:
        _temp = 'yes'
      _struct.append({item:_temp})

    ## lr
    _struct.append({'lr':_getEX(inputs.get('lr', 'default'))})

    ## flops
    _flops = inputs.get('flops')
    if _flops is not None:
      _flopsX = _getDecX(_flops)
      _struct.append({'flopsX':_flopsX})
    if _flops is None:
      _flops = 'no'
    _struct.append({'flops':_flops})

    ## parameters
    # NOTE:NTF

    ## global
    _global = []
    _global_epochs = inputs.get('global_epochs', 0)
    _global.append({'epochs':_global_epochs})
    _struct.append({'global':_global})

    return self.link('config', _struct)

  def log(self, id, meta, logs):
    """
      Log Block

      # Parm
        id: int. The serial number of the log.
        meta: dict. Contains metadata key value pairs.
        logs: dict. Contains all logs key value pairs.

      # Return
        dict
    """
    _struct = []
    _autolist = [
      'epochs',
      'enhance',
      'lr_alt',
      'time_start',
      'time_stop',
      'time_cost',
    ]
    for item in _autolist:
      _temp = meta.get(item)
      if _temp is None or _temp == False:
        _temp = 'no'
      if _temp == True:
        _temp = 'yes'
      _struct.append({item:_temp})

    ## logs part
    for inx, epoch in enumerate(logs):
      _temp_struct = []
      _temp_auto_list = [
        'train_accy',
        'train_loss',
        'train_time',
        'val_accy',
        'val_loss',
        'val_time',
      ]
      for item in _temp_auto_list:
        _temp_struct.append({item:'%.4f'%epoch[item]})
      _struct.append({f'epoch{inx+1}':_temp_struct})

    return self.link(f'log{id}', _struct)


class InfoStruct(object):
  """
    Info Struct
  """
  def __init__(self, **kwargs):
    self._struct = []
    pass

  @property
  def struct(self):
    return self._struct


# test
if __name__ == "__main__":
  _info = {
    'dataset':'cifar10',
    'lib':'standard',
    'model':'vgg16',
    'xgpu':True,
    'batchsize':128,
    'loss':'sparse_categorical_crossentropy',
    'opt':'adam',
    'lr':1e-3,
    'flops':345294400,
    'input_shape':(32,32,3),
    'global_epochs':2,}
  _log1 = {
    'epochs':2,
    'enhance':True,
    'lr_alt':False,
    'time_start':'19-10-28-10-13-00',
    'time_stop':'19-10-28-11-13-00',
    'time_cost':3600,}
  _logs = [
    {
      'train_accy':0.0398,
      'train_loss':3.7093,
      'train_time':35.0572,
      'val_accy':0.0574,
      'val_loss':2.9084,
      'val_time':0.9503,
    },
    {
      'train_accy':0.0573,
      'train_loss':3.3217,
      'train_time':35.0431,
      'val_accy':0.0574,
      'val_loss':2.9084,
      'val_time':0.9502,
    },]
  # i = _InfoStruct()
  # C = i.config(_info)
  # L = i.log(1, _log1, _logs)
  # M = i.mix(C, L)
  # i.print(M)
  i2 = InfoStruct()
  print(i2.struct)
