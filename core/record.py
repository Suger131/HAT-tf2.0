# -*- coding: utf-8 -*-
"""Record

  File:
    /hat/core/record

  Description:
    训练过程记录
"""


import gzip
import os
import sys
import pickle

import numpy as np
import tensorflow as tf
import h5py

from hat import __config__ as C
from hat import util
from hat.core import abc
from hat.core import config
from hat.core import log
from hat.model import nn


class Record(abc.Callback):
  """Record
  
    Description: 
      训练过程中更新记录文件
      生成:
        `history`文件夹: 存放中间数据和权重，格式`.gz`
        `.meta`文件: 存放元数据，包括参数和原始权重

    Args:
      None
  """
  def __init__(
      self,
      model=None,
      savedir=None,
      extra_byte=0,
      maxbyte=None,
      history_suffix=None,
      meta_suffix=None,
      **kwargs):
    self.model = model or config.get('model').model
    self.batch_size = config.get('batch_size')
    self.dir = savedir or config.get('save_dir')
    self.dir = os.path.join(self.dir, C.get('history_dir'))
    self.extra_byte = extra_byte
    self.maxbyte = maxbyte or C.get('maxbyte')
    self.suffix = history_suffix or C.get('suffix')['history']
    self.meta_suffix = meta_suffix or C.get('suffix')['meta']

    self.meta = {}
    self.len = 0
    self.batch_index = 0
    self.package = []
    self.package_index = 0
    self.train_loss = []
    self.train_accuracy = []
    del kwargs

  # public method

  def update(self, step, x, result, **kwargs):
    outputs = {}
    # step
    outputs['step'] = step
    # middle outputs
    mid_output_layers = nn.get_layer_output_name_full(self.model)
    for name in mid_output_layers:
      outputs[name + '_output'] = nn.get_layer_output(
          self.model, x, name).astype(C.get('history_dtype'))
    # middle weights
    mid_weight_layers = nn.get_layer_weight_name(self.model)
    for name in mid_weight_layers:
      outputs[name + '_weight'] = nn.get_layer_weight(
          self.model, name)
    # metrics
    outputs['loss'] = float(result[0])
    outputs['accuracy'] = float(result[1])
    self.train_loss.append(float(result[0]))
    self.train_accuracy.append(float(result[1]))
    # extra info
    outputs = {**outputs, **kwargs}

    self.write_h5(outputs)

    # self.package.append(outputs)
    # check package.len
    # if len(self.package) == self.len:
      # self.gen_package()
    # return outputs

  def summary(self):
    outputs = [
        sum(self.train_loss) / len(self.train_loss),
        sum(self.train_accuracy) / len(self.train_accuracy)]
    self.train_loss, self.train_accuracy = [], []
    return outputs

  def on_train_begin(self, learning_phase=None):
    """Called at the begin of training.
    
      Args:
        learning_phase: Int. Value must be `1` or `0`.
    """
    if learning_phase is not None:
      nn.set_learning_phase(learning_phase)
    # init_meta
    self.init_meta()

  def on_train_end(self, learning_phase=None):
    """Called at the end of training.
    
      Args:
        learning_phase: Int. Value must be `1` or `0`.
    """
    if learning_phase is not None:
      nn.set_learning_phase(learning_phase)
    # check if package not empty
    # if self.package:
    #   self.gen_package()
    # end_meta

  # built-in method

  def init_meta(self):
    K = tf.keras.backend
    input_shape = (1,) + K.int_shape(self.model.input)[1:]
    x = K.zeros(input_shape)

    took_byte = 0
    # step
    took_byte += sys.getsizeof(0)
    # middle outputs
    mid_output_layers = nn.get_layer_output_name_full(self.model)
    for name in mid_output_layers:
      took_byte += sys.getsizeof(nn.get_layer_output(self.model, x,
          name).astype(C.get('history_dtype'))) * self.batch_size
    # middle weights
    mid_weight_layers = nn.get_layer_weight_name(self.model)
    for name in mid_weight_layers:
      for w in nn.get_layer_weight(self.model, name):
        took_byte += sys.getsizeof(w)
    # metrics
    took_byte += sys.getsizeof(float(0)) * 2
    # extra info
    took_byte += self.extra_byte
    # fix gzip
    self.len = self.maxbyte // (took_byte / 4 * 3)

    log.info(f'package max steps: {self.len}', name=__name__)
    log.info(f'package bytes: {took_byte}', name=__name__)
    # init weights
    init_weights = {}
    for name in mid_weight_layers:
      init_weights[name + '_weight'] = nn.get_layer_weight(
          self.model, name)
    self.meta['init_weights'] = init_weights
    # configs
    self.meta['batch_size'] = config.get('batch_size')
    self.meta['history_dir'] = self.dir
    log.debug(self.meta, name=__name__)
    if not os.path.exists(self.dir):
      os.makedirs(self.dir)

  def gen_package(self):
    p_name = str(self.package_index) + self.suffix
    log.info(f'Saving package: {p_name}', name=__name__)
    with gzip.open(os.path.join(self.dir, p_name), 'wb') as f:
      pickle.dump(self.package, f)
    self.package = []
    self.package_index += 1

  def write_h5(self, batch:dict):
    package_index = self.batch_index // self.len
    h5_name = str(package_index) + self.suffix
    with h5py.File(os.path.join(self.dir, h5_name), 'a') as hf:
      subgroup = hf.create_group(str(self.batch_index))
      for item in batch:
        if isinstance(batch[item], np.ndarray):
          subgroup.create_dataset(item, data=batch[item],
              compression='gzip', compression_opts=5)
        elif isinstance(batch[item], list):
          ssub = subgroup.create_group(item)
          for inx, data in enumerate(batch[item]):
            ssub.create_dataset(str(inx), data=data,
                compression='gzip', compression_opts=5)
        else:
          subgroup.create_dataset(item, data=np.atleast_1d(batch[item]))
      # hf.create_dataset(str(self.batch_index), data=batch)
    self.batch_index += 1


# test
if __name__ == "__main__":
  from hat.app.standard import mlp
  log.init('./unpush/test', detail=True)
  config.test((28, 28, 1), (10,))
  mod = mlp.mlp()
  mod.summary()
  # config.set('model', mod)
  r = Record(mod.model, './unpush')
  # print(r.get_size())

