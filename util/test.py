# -*- coding: utf-8 -*-
"""Test

  File: 
    /hat/util/test

  Description: 
    test tools
"""


# import setting
__all__ = [
    'TestConfig',]


from hat.util import abc


class TestConfig(object):
  """TestConfig
  
    Description: 
      用于测试的Config类，能存放参数

    Args:
      **kwargs: Any. 任意参数，自动加入到属性中
  """
  def __init__(self, **kwargs):
    self.__dict__ = {**self.__dict__, **kwargs}
    self.data = abc.Empty()
    self.dtype = 'float32'
    self.xgpu = False
    self.batch_size = 128
    self.epochs = 5
    self.opt = None
    self.load_name = ''


# test part
if __name__ == "__main__":
  from hat.util import log
  from hat.util import util
  tc = TestConfig(name='test')
  log.init('./unpush/test')
  log.log(tc.name)
  util.del_tail_digit('vgg16')
  # print(tc.name)
