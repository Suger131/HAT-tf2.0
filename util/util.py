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
]


def del_tail_digit(inputs):
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

