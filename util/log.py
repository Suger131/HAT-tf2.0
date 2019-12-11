# -*- coding: utf-8 -*-
"""Log

  File: 
    /hat/util/log

  Description: 
    log tools
"""


# import setting
__all__ = [
  
]


import logging


class Logger(object):
  """Logger
  
    Description: 
      日志工具

    Attributes:
      log_dir: Str. 日志目录
  """
  def __init__(self, log_dir):
    self.log_dir = log_dir

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)

    log.info('Logger is ready.')


# test part
if __name__ == "__main__":
  log1 = Logger('./')
  from hat.util import test
  tc = test.TestConfig()



