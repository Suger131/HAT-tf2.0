"""
  hat.utils.logger
"""


__all__ = [
  'logger'
]


import os
import time


def _writer(func):
  def _inner(self, *args, **kwargs):
    with open(self._log_dir, 'a+') as f:
      for i in args: f.write(i + '\n')
    func(self, *args, **kwargs)
  return _inner


class logger(object):
  """
    logger
  """
  def __init__(self, log_dir, log_name=None, suffix='.txt', **kwargs):
    if not log_dir: raise Exception(f"log_dir must not be a null value.")
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    self._log_name = log_name or self._time
    self._suffix = suffix
    self._log_dir = f'{log_dir}/{self._log_name}{self._suffix}'
    self._call(f'Logger has been Loaded.')

  def __call__(self, *args, **kwargs):
    return self._call(*args, **kwargs)

  def _call(self, *args, **kwargs):
    _args, _form = self._log(*args, **kwargs)
    for i in _args:
      self._print(f"{_form} {i}")
    return True

  def _log(self, *args, **kwargs):
    # processing args
    _args = self._p_args(args)
    # processing kwargs
    _form = self._p_kwargs(kwargs)
    return _args, _form

  @_writer
  def _print(self, value):
    print(value)

  @property
  def _time(self):
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

  def _p_args(self, args):
    _args = []
    [_args.extend(i) if type(i) in [list, tuple] else _args.append(i) for i in args]
    return _args
  
  def _p_kwargs(self, kwargs):
    _addition = ''
    _text = ''
    _variable = None
    _list = []
    _dict = {}
    _boor = None

    for i in ['text', 't']:
      if i in kwargs: _text = kwargs.pop(i)
    for i in ['addition', 'a']:
      if i in kwargs: _addition = kwargs.pop(i)
    for i in ['variable', 'var', 'v']:
      if i in kwargs: _variable = kwargs.pop(i)
    for i in ['list', 'l']:
      if i in kwargs: _list = kwargs.pop(i)
    for i in ['dict', 'd']:
      if i in kwargs: _dict = kwargs.pop(i)
    for i in ['boor', 'b']:
      if i in kwargs: _boor = kwargs.pop(i)
    if '' in kwargs:
      kwargs.pop('')

    # processing V and L&D
    if _variable:
      if _list:
        if _variable not in _list:
          return
      if _dict:
        if _variable not in _dict:
          return
        else:
          _text = _dict[_variable]
    if _boor != None:
      if _list and len(_list) == 2:
        _text = _boor and _list[0] or _list[1]
      else:
        raise Exception(f"Args missing, list must have 2 Args, but got '{len(_list)}'")

    _form = []
    _form.append(f'[{self._time}]')
    _form.append(f' [LOG]')
    if _addition: _form.append(f' [{_addition}]')
    if _text: _form.append(f' {_text}')
    return f''.join(_form)

  # public method

  def error(self, args, text=''):
    self._call(args, t=text, a='Error')
    os._exit(1)


if __name__ == "__main__": 
  a = logger(log_dir='./')
  a(['/', '123'], text='Unsupported option:', _A='Warning')
  a(200, _V='', _L=['', 'train'], _T='Epochs:')
  a('', _L=['Yes', 'No'], _B=1 == 2)
