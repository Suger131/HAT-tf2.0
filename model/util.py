# -*- coding: utf-8 -*-
"""utils

  File: 
    /hat/model/custom/util

  Description: 
    util
"""


def int_to_tuple(element: int, length: int):
  """整数转元组

    Description:
      自动将Int转为Tuple，适用于用一个整数表示元组中所有元素均为该整数值
      的情况。

    Args:
      element: Int. 传入的整数
      length: Int. 元组长度

    Returns:
      Tuple
    
    Raises:
      None
  """
  return (element,) * length


def normalize_tuple(obj, length: int):
  """标准化为元组

    Description:
      将tuple/list/int标准化为元组，并且符合指定长度

    Args:
      obj: Tuple/List/Int. 传入的对象
      length: Int. 元组长度

    Returns:
      Tuple
    
    Raises:
      TypeError
      LenError
  """
  assert isinstance(obj, (tuple, list, int)), f'[TypeError] obj ' \
         f'must be tuple/list/int, but got {type(obj).__name__}. '
  new_tuple = ()
  if isinstance(obj, (tuple, list)):
    if len(obj) == length:
      new_tuple = tuple(obj)
    elif len(obj) == 1:
      new_tuple = int_to_tuple(obj[0], length)
    else:
      raise Exception(f'[LenError] obj({type(obj).__name__}).len '
            f'must be 1 or {length}, but got {len(obj)}. ')
  else:
    new_tuple = int_to_tuple(obj, length)
  return new_tuple


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  """计算卷积输出尺寸

    Description:
      确定给定输入长度的卷积的输出长度。

    Args:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full", "causal"
      stride: integer.
      dilation: dilation rate, integer.

    Returns:
      The output length (integer).
    
    Raises:
      None
  """
  if input_length is None:
    return None
  assert padding in {'same', 'valid', 'full', 'causal'}
  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  if padding in ['same', 'causal']:
    output_length = input_length
  elif padding == 'valid':
    output_length = input_length - dilated_filter_size + 1
  elif padding == 'full':
    output_length = input_length + dilated_filter_size - 1
  return (output_length + stride - 1) // stride


# test part
if __name__ == "__main__":
  print(normalize_tuple((1,1), 2))


