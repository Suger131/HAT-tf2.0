import os
import time

from utils._func import _writer


class Log(object):
  """
  Logger
  """

  def __init__(self, log_dir, log_name=None, ext='.txt'):
    self._ext = ext
    if not log_dir: raise Exception(f"log_dir must not be a null value.")
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    self._log_name = log_name or self._time
    self._log_dir = f'{log_dir}/{self._log_name}{self._ext}'
    self._log(f'Logger has been Loaded.')
  
  def __call__(self, *args, **kwargs):
    _args, _form = self._log(*args, **kwargs)
    # log
    for i in _args:
      self._print(f"{_form} {i}")
    return True

  def _w(self, *args, **kwargs):
    _args, _form = self._log(*args, **kwargs)
    for i in _args:
      self._write(f"{_form} {i}")
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

  @_writer
  def _write(self, value):
    pass

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

    for i in ['text', '_T']:
      if i in kwargs: _text = kwargs.pop(i)
    for i in ['addition', '_A']:
      if i in kwargs: _addition = kwargs.pop(i)
    for i in ['variable', 'var', '_V']:
      if i in kwargs: _variable = kwargs.pop(i)
    for i in ['list', '_L']:
      if i in kwargs: _list = kwargs.pop(i)
    for i in ['dict', '_D']:
      if i in kwargs: _dict = kwargs.pop(i)
    for i in ['boor', '_B']:
      if i in kwargs: _boor = kwargs.pop(i)
    if '' in kwargs:
      kwargs.pop('')

    # processing _V and _L&_D
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


if __name__ == "__main__": 
  a = Log(log_dir='./')
  a(['/', '123'], text='Unsupported option:', _A='Warning')
  a(200, _V='', _L=['', 'train'], _T='Epochs:')
  a('', _L=['Yes', 'No'], _B=1 == 2)
