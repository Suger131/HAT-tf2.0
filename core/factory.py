# -*- coding: utf-8 -*-
"""Facroty

  File:
    /hat/core/facroty

  Description:
    模型核心代码
    包括模型图生成、训练、验证、预测、保存等
"""


import os

import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard

from hat import __config__ as C
from hat import util
from hat.core import config
from hat.core import log
from hat.core import record
from hat.dataset import generator


class Factory(object):
  """Factory
  
    Description: 
      训练工具

    Args:
      None

    Method:
      run
      train
      val
      save
  """
  def __init__(self, *args, **kwargs):
    self.model = None
    self.data = None
    self.batch_size = 0
    self.epochs = 0
    self.step = 0
    self.record = None
    del args, kwargs
  
  # Built-in method

  def _datagen(self):
    """Image Data Enhancement"""
    return config.get('aug').flow(
        self.data.train_x,
        self.data.train_y,
        batch_size=self.batch_size,
        shuffle=True)

  # Public method

  def init(self):
    self.model = config.get('model').model
    self.data = config.get('data')
    self.batch_size = config.get('batch_size')
    self.epochs = config.get('epochs')
    self.step = config.get('step')
    if self.step:
      self.record = record.Record()
    else:
      self.record = record.Record(extra_byte=len(bytes(0))) # extra info is epoch

  def run(self):
    try:
      self.get_image()
      self.train()
      self.val()
      self.test()
      self.save()
    except Exception:
      # In case of unknown errors
      log.exception(name=__name__)

  def train(self):
    if not config.get('is_train'):
      return
    def inner_train():
      log.info(f"Train Start", name=__name__)
      dg = generator.Generator(
          self.data.train_x,
          self.data.train_y,
          self.batch_size)
      def train_core(step, max_step, epoch=None):
        self.record.on_batch_begin()
        train_x, train_y = next(dg)
        result = self.model.train_on_batch(train_x, train_y)
        if epoch:
          self.record.on_batch_end(step, train_x, result, epoch=epoch)
        else:
          self.record.on_batch_end(step, train_x, result)
        if step % config.get('step_per_log') == 0 or step == max_step:
          results = self.record.summary()
          step_log = [
              f'Step: {step}/{max_step}, ',
              f'loss: {results[0]:.4f}, ',
              f'accuracy: {results[1]:.4f}']
          if epoch:
            step_log = [f'Epoch: {epoch}, '] + step_log
          log.info(''.join(step_log), name=__name__)
      
      if self.step:
        self.record.on_train_begin(learning_phase=1)
        for i in range(1, self.step+1):
          train_core(i, self.step)
          if i % config.get('step_per_val') == 0 and i != self.step:
            # log.info(f"Step: {i} Val", name=__name__)
            val_result = self.model.evaluate(
                self.data.val_x,
                self.data.val_y,
                batch_size=self.batch_size,
                verbose=0)
            log.info(f"Step: {i} Val, loss: {val_result[0]:.4f}" \
                f", accuracy: {val_result[1]:.4f}", name=__name__)
        log.info(f"Step Over.", name=__name__)
        self.record.on_train_end(learning_phase=0)
      else:
        self.record.on_train_begin()
        for ep in range(self.epochs):
          self.record.on_epoch_begin()
          log.info(f"Epoch: {ep+1}/{self.epochs} Train",
              name=__name__)
          for i in range(dg.len):
            train_core(i+1, dg.len)
          log.info(f"Epoch: {ep+1}/{self.epochs} Val",
              name=__name__)
          val_result = self.model.evaluate(
              self.data.val_x,
              self.data.val_y,
              batch_size=self.batch_size,
              verbose=0)
          log.info(f"Epoch: {ep+1}/{self.epochs}, " \
              f"accuracy: {val_result[1]:.4f}, " \
              f"loss: {val_result[0]:.4f}",
              name=__name__)
          self.record.on_epoch_end()
        self.record.on_train_end()
      log.info(f"Train Stop", name=__name__)
    cost_time = util.get_cost_time(inner_train)[0]
    log.info(f"Train cost time: {cost_time}", name=__name__)

  def val(self):
    if not config.get('is_val'):
      return
    def inner_val():
      log.info(f"Val Start", name=__name__)
      result = self.model.evaluate(
          self.data.val_x,
          self.data.val_y,
          batch_size=self.batch_size,
          verbose=0)
      log.info(f"Total Loss: {result[0]:.4f}", name=__name__)
      log.info(f"Accuracy: {result[1]:.4f}", name=__name__)
      log.info(f"Val Stop", name=__name__)
    cost_time = util.get_cost_time(inner_val)[0]
    log.info(f"Val cost time: {cost_time}", name=__name__)

  def test(self):
    if not config.get('is_test'):
      return

  def save(self):
    if not config.get('is_save'):
      return
    self.model.save(config.get('save_name'))
    log.info(f"Saved model: {config.get('save_name')}",
        name=__name__)

  def get_image(self):
    if not config.get('is_gimage'):
      return
    img_name = config.get('log_dir') + '.png'
    if os.path.exists(img_name):
      return
    try:
      tf.compat.v1.keras.utils.plot_model(
          self.model,
          to_file=img_name,
          # dpi=64,
          show_shapes=True,)
    except Exception:
      # pydot bug
      log.exception(name=__name__)
    log.info(f"Generated model img: {img_name}", name=__name__)


factory = Factory()


def init():
  factory.init()


def run():
  factory.run()


def train():
  factory.train()


def val():
  factory.val()


def test():
  factory.test()


def get_image():
  factory.get_image()


def save():
  factory.save()


# test
if __name__ == "__main__":
  pass

