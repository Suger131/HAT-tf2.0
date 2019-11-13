"""
  hat.utils._TC

  test config
"""

# pylint: disable=no-name-in-module


__all__ = [
  '_TC'
]


class _TC(object):
  """
    _TC

    config object for test
  """
  def __init__(self, *args, **kwargs):
    self.dtype = 'float32'
    self.xgpu = False
    self.batch_size = 128
    self.epochs = 5
    self.opt = None

