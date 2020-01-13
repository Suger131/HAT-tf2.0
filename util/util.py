# -*- coding: utf-8 -*-
"""util

  File: 
    /hat/util/util

  Description: 
    Common tools
"""


# import setting
__all__ = [
    'del_tail_digit',
    'get_iuwhx',
    'get_ex',
    'get_cost_time',]


import time


def quadrature_list(iterable):
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


def get_cost_time(func, *args, **kwargs):
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

