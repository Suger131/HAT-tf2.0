# -*- coding: utf-8 -*-
"""Log

  File: 
    /hat/core/log

  Description: 
    logger
"""


import os
import logging
from logging import handlers


_BUILT = False
_DETAIL = False
_LOGGING_FUNCTION = [
    logging.debug,
    logging.info,
    logging.warning,
    logging.error,
    logging.fatal,
    logging.exception,]
_LEVEL_LIST = [
    'debug',
    'info',
    'warn',
    'error',
    'fatal',]


def _check_built(func):
  def log_function(*args, **kwargs):
    if _BUILT:
      return func(*args, **kwargs)
    raise Exception('[InitError] hat.util.log has not yet '\
        f'been initialized')
  return log_function


def init(log_dir, suffix='.log', filemode='a+', detail=False):
  """log.init

    Description: 
      初始化日志工具，在第一次调用日志输出函数前必须调用本函数
      日志同时输出到控制台和日志文件
      日志等级有：INFO、WARN、ERROR、FATAL。另外有DEBUG(开启
      detail模式可用)

    Args:
      log_dir: Str. 日志目录
      suffix: Str, default '.log'. 日志后缀名
      filemode: Str, default 'a'. 日志文件打开方式，同open.mode
      detail: Bool, default False. 日志输出粒度。默认最低从INFO
          开始，若选择detail模式，日志文件写入最低改为DEBUG。

    Return:
      None

    Raises:
      None
  """
  global _BUILT, _DETAIL
  formatter = logging.Formatter(
      fmt='%(asctime)s.%(msecs)03d [%(levelname)s] >%(name)s: %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S')
  handler = handlers.RotatingFileHandler(
      log_dir + suffix,
      mode=filemode,
      maxBytes=2 ** 25,
      backupCount=1000)
  if detail:
    handler.setLevel(logging.DEBUG)
    _DETAIL = detail
  else:
    handler.setLevel(logging.INFO)
  handler.setFormatter(formatter)
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  console.setFormatter(formatter)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  logger.addHandler(handler)
  logger.addHandler(console)
  _BUILT = True

@_check_built
def debug(msg, name=None, **kwargs):
  """log.debug

    Description: 
      DEBUG级别的日志，只有开启detail模式才有效

    Args:
      msg: Str. 日志内容
      name: Str, default None. 默认为root，可用于定位日志的发送位置
      **kwargs: 参见logging.debug

    Return:
      None

    Raises:
      None
  """
  if _DETAIL:
    logging.getLogger(name).debug(msg, **kwargs)

@_check_built
def info(msg, name=None, **kwargs):
  """log.info

    Description: 
      INFO级别的日志

    Args:
      msg: Str. 日志内容
      name: Str, default None. 默认为root，可用于定位日志的发送位置
      **kwargs: 参见logging.info

    Alias:
      log

    Return:
      None

    Raises:
      None
  """
  logging.getLogger(name).info(msg, **kwargs)

@_check_built
def warn(msg, name=None, **kwargs):
  """log.warn

    Description: 
      WARN级别的日志

    Args:
      msg: Str. 日志内容
      name: Str, default None. 默认为root，可用于定位日志的发送位置
      **kwargs: 参见logging.warning

    Return:
      None

    Raises:
      None
  """
  logging.getLogger(name).warning(msg, **kwargs)

@_check_built
def error(msg, name=None, exit=False, **kwargs):
  """log.error

    Description: 
      ERROR级别的日志

    Args:
      msg: Str. 日志内容
      name: Str, default None. 默认为root，可用于定位日志的发送位置
      exit: Bool, default False. 是否在记录日志后退出程序，默认不退出
      **kwargs: 参见logging.error

    Return:
      None

    Raises:
      None
  """
  logging.getLogger(name).error(msg, **kwargs)
  if exit:
    os._exit(0)

@_check_built
def fatal(msg, name=None, exit=False, **kwargs):
  """log.fatal

    Description: 
      FATAL级别的日志

    Args:
      msg: Str. 日志内容
      name: Str, default None. 默认为root，可用于定位日志的发送位置
      exit: Bool, default False. 是否在记录日志后退出程序，默认不退出
      **kwargs: 参见logging.fatal

    Return:
      None

    Raises:
      None
  """
  logging.getLogger(name).fatal(msg, **kwargs)
  if exit:
    os._exit(0)

@_check_built
def exception(msg='Exception Logged', name=None, exit=False, **kwargs):
  """log.exception

    Description: 
      记录异常日志，日志等级为ERROR

    Args:
      msg: Str, default 'Exception Logged'. 提示捕抓异常日志的语句
      name: Str, default None. 默认为root，可用于定位日志的发送位置
      exit: Bool, default False. 是否在记录日志后退出程序，默认不退出
      **kwargs: 参见logging.fatal

    Return:
      None

    Raises:
      None
  """
  logging.getLogger(name).exception(msg, **kwargs)
  if exit:
    os._exit(0)

@_check_built
def log_list(inputs, level='info', name=None, prefix='', suffix='', **kwargs):
  """log.log_lsit

    Description: 
      批量log

    Args:
      inputs: List of Str. 列表形式的日志内容
      level: Str, default 'info'. 日志等级，默认为INFO
      name: Str, default None. 默认为root，可用于定位日志的发送位置
      prefix: Str, default ''. 前缀日志内容
      suffix: Str, default ''. 后缀日志内容
      **kwargs: 参见logging.info

    Return:
      None

    Raises:
      None
  """
  _repacked_function = [
      debug,
      info,
      warn,
      error,
      fatal,]
  _level_to_function = dict(zip(_LEVEL_LIST, _repacked_function))
  if level not in _LEVEL_LIST:
    level = 'info'
  log_func = _level_to_function[level]
  for msg in inputs:
    log_func(prefix + msg + suffix, name=name, **kwargs)


# Alias
log = info


# test part
if __name__ == "__main__":
  init('./unpush/test')
  info("Logger is ready", 'hat')
  warn("Check name.", __name__)
  log_list(['x=3', '-Y'], prefix='Unsupported option: ', level='warn')
  error("vgg16 is not in the lib", 'hat.util.importer')
  fatal("Low battery!")

