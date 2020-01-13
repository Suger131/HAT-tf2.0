# -*- coding: utf-8 -*-
"""Util

  File: 
    /hat/dataset/util

  Description: 
    utils
"""


from random import shuffle as _shuffle


def shuffle(inputs, islist=True):
  """shuffle

    Description:
      Shuffle the datas/labels
      Private method

    Args:
      inputs: np.array/list, or list/tuple of np.array/list. 
      In the latter case, the corresponding elements for each 
      item in the list are shuffled together as a whole.
      islist: Boolean. The latter case is enabled when the 
      value is True. Default is True.

    Return:
      A shuffled np.array/list(depend on argus)
  """
  if islist:
    len_ = len(inputs[0])
    index = list(range(len_))
    _shuffle(index)
    outputs = [[i[index[j]] for j in range(len_)] for i in inputs]
  else:
    len_ = len(inputs)
    index = list(range(len_))
    _shuffle(index)
    outputs = [inputs[i] for i in index]
  return outputs




