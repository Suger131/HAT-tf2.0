# -*- coding: utf-8 -*-
"""Facroty

  File:
    /hat/core/facroty

  Description:
    模型核心代码
    包括模型图生成、训练、验证、预测、保存等
"""


import gzip
import os
import pickle
import csv
import codecs

import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard

from hat import __config__ as C
from hat import util
from hat.core import log
from hat.dataset import generator
from hat.model import nn


class _Store(object):
  """Store Variable"""
  def __init__(self):
    self.train_loss = []
    self.train_accuracy = []
    # self.init()

  def init(self):
    # self.train_loss = 0
    # self.train_accuracy = 0
    # self.count = 0
    pass

  def get_summary(self):
    outputs = [
        sum(self.train_loss) / len(self.train_loss),
        sum(self.train_accuracy) / len(self.train_accuracy)]
    self.train_loss, self.train_accuracy = [], []
    # outputs = [self.train_loss / self.count,
    #            self.train_accuracy / self.count]
    # self.init()
    return outputs

  def update_train(self, result):
    self.train_loss.append(result[0])
    self.train_accuracy.append(result[1])
    # self.train_loss += result[0]
    # self.train_accuracy += result[1]
    # self.count += 1

  def norm(self, result):
    return f'{result[0]:.4f}', f'{result[1]:.4f}'


class _Record(object):
  """Record"""
  def __init__(
      self,
      model,
      batch_size,
      save_dir,
      extra_byte=0,
      maxbyte=C.get('maxbyte'),
      history_suffix=C.get('suffix')['history'],
      meta_suffix=C.get('suffix')['meta'],
      **kwargs):
    self.model = model
    self.batch_size = batch_size
    self.dir = save_dir
    self.extra_byte = extra_byte
    self.maxbyte = maxbyte
    self.suffix = history_suffix
    self.meta_suffix = meta_suffix

    self.len = maxbyte // self.get_size()
    self.package = []
    self.package_index = 0
    
    self.train_loss = []
    self.train_accuracy = []
    del kwargs

  def get_size(self):
    K = tf.keras.backend
    input_shape = (1,) + K.int_shape(self.model.input)[1:]
    x = K.zeros(input_shape)
    layers_name = nn.get_layer_output_name_full(self.model)
    weight_name = nn.get_layer_weight_name(self.model)
    took_byte = 0
    # output
    outputs = []
    for name in layers_name:
      outputs.append(nn.get_layer_output(self.model, x, name))
    took_byte += sum([util.quadrature_list(i.shape) for i in outputs]) * self.batch_size
    # weight
    outputs = []
    for name in weight_name:
      outputs.append(nn.get_layer_weight(self.model, name))
    # the weight type is `list of np.ndarray`
    took_byte += sum([util.quadrature_list(j.shape) for i in outputs for j in i])
    # metrics, included loss and accuracy
    took_byte += 2
    # extra info
    took_byte += self.extra_byte
    return took_byte

  def get_summary(self):
    outputs = [
        sum(self.train_loss) / len(self.train_loss),
        sum(self.train_accuracy) / len(self.train_accuracy)]
    self.train_loss, self.train_accuracy = [], []
    return outputs

  def update(self, step, x, result, **kwargs):
    outputs = {}
    # step
    outputs['step'] = step
    # middle outputs
    mid_output_layers = nn.get_layer_output_name_full(self.model)
    # mid_outputs = []
    for name in mid_output_layers:
      outputs[name + '_output'] = nn.get_layer_output(
          self.model, x, name)
      # mid_outputs.append(nn.get_layer_output(self.model, x, name))
    # middle weights
    mid_weight_layers = nn.get_layer_weight_name(self.model)
    # mid_weights = []
    for name in mid_weight_layers:
      outputs[name + '_weight'] = nn.get_layer_weight(
          self.model, name)
      # mid_weights.append(nn.get_layer_weight(self.model, name))
    # metrics
    outputs['loss'] = result[0]
    outputs['accuracy'] = result[1]
    self.train_loss.append(result[0])
    self.train_accuracy.append(result[1])
    # extra info
    outputs = {**outputs, **kwargs}

    self.package.append(outputs)
    # check package.len
    if len(self.package) == self.len:
      self.gen_package()
    # return outputs

  def gen_package(self):
    if not self.package:
      return
    log.info(f'Saving package: {self.package_index}{self.suffix}', 
        name=__name__)
    with gzip.open(os.path.join(
        self.dir,
        str(self.package_index) + self.suffix), 'wb') as f:
      pickle.dump(self.package, f)
    self.package = []
    self.package_index += 1


class Factory(object):
  """Factory
  
    Description: 
      训练工具

    Args:
      config: hat.Config.

    Method:
      run
      train
      val
      save

    Usage:
    ```python
      import hat
      F = hat.Factory(C)
      # or
      F = hat.core.Factory(C)
      # or
      F = hat.core.factory.Factory(C)
      # or
      from hat.core import Factory
      F = Factory(C)
    ```
  """
  def __init__(self, config, *args, **kwargs):
    self.config = config
    self.model = config.model.model
    self._store = _Store()
    self._record = _Record(self.model, self.config.batch_size,
        self.config.save_dir)
    log.info(f'package size: {self._record.len}', name=__name__)
  # Built-in method

  def _datagen(self):
    """Image Data Enhancement"""
    return self.config.aug.flow(
        self.config.data.train_x,
        self.config.data.train_y,
        batch_size=self.config.batch_size,
        shuffle=True)

  # Public method

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
    if not self.config.is_train:
      return
    def inner_train():
      log.info(f"Train Start", name=__name__)
      dg = generator.Generator(
          self.config.data.train_x,
          self.config.data.train_y,
          self.config.batch_size)
      mid_output_layers = nn.get_layer_output_name(self.model)
      mid_weight_layers = nn.get_layer_weight_name(self.model)
      def train_core(step, max_step):
        # train_x, train_y = dg.__getitem__(0)
        train_x, train_y = next(dg)
        result = self.model.train_on_batch(train_x, train_y)
        # _time, result = util.get_cost_time(
        #     self.model.train_on_batch,
        #     train_x,
        #     train_y)
        # if self.config.is_write_middle_data:
        #   for j in mid_output_layers:
        #     mid_output = nn.get_layer_output(self.model, train_x, j)
        #     filename = f'{self.config.save_dir}/middle_output_{j}.csv'
        #     with codecs.open(filename, 'a+', 'utf-8') as f:
        #       writer = csv.writer(f, dialect='excel')
        #       writer.writerow(mid_output.tolist())
        #   for j in mid_weight_layers:
        #     mid_weight = nn.get_layer_weight(self.model, j)
        #     filename = f'{self.config.save_dir}/middle_weight_{j}.csv'
        #     with codecs.open(filename, 'a+', 'utf-8') as f:
        #       writer = csv.writer(f, dialect='excel')
        #       writer.writerow([z.tolist() for z in mid_weight])
        # if self.config.is_write_loss:
        #   filename = f"{self.config.save_dir}/loss.csv"
        #   with codecs.open(filename, 'a+', 'utf-8') as f:
        #     writer = csv.writer(f, dialect='excel')
        #     # writer.writerow(result)
        #     writer.writerow(self._store.norm(result))
        
        # self._store.update_train(result)
        self._record.update(step, train_x, result)
        if step % self.config.step_per_log == 0 or step == max_step:
          # results = self._store.get_summary()
          results = self._record.get_summary()
          log.info(
              f'Step: {step}/{max_step}, ' \
              f'loss: {results[0]:.4f}, ' \
              f'accuracy: {results[1]:.4f}',
              name=__name__)
      
      if self.config.step:
        tf.keras.backend.set_learning_phase(1)
        for i in range(self.config.step):
          train_core(i + 1, self.config.step)
        log.info(f"Step Over.", name=__name__)
        self._record.gen_package()
        tf.keras.backend.set_learning_phase(0)
      else:
        for ep in range(self.config.epochs):
          log.info(f"Epoch: {ep+1}/{self.config.epochs} Train",
              name=__name__)
          for i in range(dg.len):
            train_core(i+1, dg.len)
          # self.config.model.fit(
          #     self.config.data.train_x,
          #     self.config.data.train_y,
          #     self.config.batch_size,
          #     epochs=1)
          log.info(f"Epoch: {ep+1}/{self.config.epochs} Val",
              name=__name__)
          val_result = self.config.model.evaluate(
              self.config.data.val_x,
              self.config.data.val_y,
              batch_size=self.config.batch_size,
              verbose=0)
          log.info(f"Epoch: {ep+1}/{self.config.epochs}, " \
              f"accuracy: {val_result[1]:.4f}, " \
              f"loss: {val_result[0]:.4f}",
              name=__name__)
      log.info(f"Train Stop", name=__name__)
    cost_time = util.get_cost_time(inner_train)[0]
    log.info(f"Train cost time: {cost_time}", name=__name__)

  def val(self):
    if not self.config.is_val:
      return
    def inner_val():
      log.info(f"Val Start", name=__name__)
      result = self.config.model.evaluate(
          self.config.data.val_x,
          self.config.data.val_y,
          batch_size=self.config.batch_size,
          verbose=0)
      log.info(f"Total Loss: {result[0]:.4f}", name=__name__)
      log.info(f"Accuracy: {result[1]:.4f}", name=__name__)
      log.info(f"Val Stop", name=__name__)
    cost_time = util.get_cost_time(inner_val)[0]
    log.info(f"Val cost time: {cost_time}", name=__name__)

  def test(self):
    if not self.config.is_test:
      return

  def save(self):
    if not self.config.is_save:
      return
    self.config.model.save(self.config.save_name)
    log.info(f"Saved model: {self.config.save_name}",
        name=__name__)

  def get_image(self):
    if not self.config.is_gimage:
      return
    img_name = self.config.log_dir + '.png'
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

  


