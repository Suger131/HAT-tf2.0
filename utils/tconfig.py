"""
  hat.utils.tconfig
"""

# pylint: disable=no-name-in-module


__all__ = [
  'tconfig'
]


class tconfig(object):
  """
    tconfig

    config object for test
  """
  def __init__(self, *args, **kwargs):
    self.dtype = 'float32'
    self.xgpu = False
    self.batch_size = 128
    self.epochs = 5
    self.opt = None

