# -*- coding: utf-8 -*-
"""Util

  File: 
    /hat/util

  Description: 
    utils
"""


import time

from hat.core import abc


def quadrature_list(iterable) -> int:
  if not iterable:
    return 0
  if isinstance(iterable, tuple):
    iterable = list(iterable)
  if len(iterable) == 1:
    return iterable.pop()
  else:
    return iterable.pop() * quadrature_list(iterable)


def del_tail_digit(inputs) -> str:
  """Delete Tail Digit

    Description: 
      删除字符串尾部的数字

    Args:
      inputs: Str. 目标字符串

    Return:
      Str

    Raises:
      None
  """
  inx = 0
  for i in inputs:
    if i.isdigit():
      break
    inx += 1
  if inx != len(inputs):
    inputs = inputs[:inx]
  return inputs


def get_iuwhx(x: int, rounder=2) -> str:
  """Get Number With International Unit Word Head

    Description: 
      获取某一数字的带国际单位制词头的表示形式

    Args:
      x: Int. 目标数字
      rounder: Int, default 2. 小数点后几位，默认为2

    Return:
      Str

    Raises:
      None
  """
  _Xlen = (len(str(int(x))) - 1) // 3
  _X = 'KMGTPEZY'[_Xlen - 1]
  _num = round(x / (10 ** (_Xlen * 3)), rounder)
  return f'{_num}{_X}'


def get_ex(x: float) -> str:
  """Get Scientific Counting Number

    Description: 
      获取某一数字的科学计数法的表示形式

    Args:
      x: float. 目标数字

    Return:
      Str

    Raises:
      None
  """
  return f"{x:.1e}"


def get_cost_time(func, *args, **kwargs) -> [str, list]:
  """Get Function Cost Time

    Description: 
      获取函数的运行时间，精确到毫秒

    Args:
      func: Function. 目标函数
      *args, **kwargs: 函数需要的参数

    Return:
      Str of run time
      Function.Returns

    Raises:
      None
  """
  t1 = time.perf_counter()
  result = func(*args, **kwargs)
  t2 = time.perf_counter()
  return f"{t2 - t1:.3f}", result


class Counter(object):
  """Counter 
    
    Description: 
      全局计数器，根据名字自动计数

    Args:
      name: Str. 计数器名字

    Return:
      Str number.

    Usage:
      The `Counter` can be used as a `str` object.
      Or like a function. 
      This two method use the `default initial value 1`.
      And each visit will automatically increase by 1.
      Or use the `get/set` method. 
      The `set` method can use a free initial value.
      The `get` method do not change the value.

    Example:
    ```python
      x = f"{Counter('a')}"
      print(Counter('a'))
      Counter()('a')

      a=Counter()
      a.set('b', 5)
      a.get('b')
    ```
  """
  count = {}

  def __init__(self, name=None):
    self.name = name

  def __str__(self):
    if self.name:
      if self.name not in Counter.count:
        Counter.count[self.name] = 1
      else:
        Counter.count[self.name] += 1
      return str(Counter.count[self.name])
    return None
  
  def __call__(self, name):
    if name:
      if name not in Counter.count:
        Counter.count[name] = 1
      else:
        Counter.count[name] += 1
      return str(Counter.count[name])
    return None

  def get(self, name):
    """Get the count number but not change.

      Args:
        name: Str. 计数器名字
    """
    if name in Counter.count:
      return str(Counter.count[name])
    return None
      
  def set(self, name, value):
    """Set the count number.

      Args:
        name: Str. 计数器名字
        value : Int. 计数器的值
    """
    if type(value) != int:
      raise Exception(f"Value must be a int, but got {type(value).__name__}")
    Counter.count[name] = value

