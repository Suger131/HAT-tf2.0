# -*- coding: utf-8 -*-
"""Record

  File:
    /hat/core/record

  Description:
    训练过程记录
"""


import os
import sys

import numpy as np
import tensorflow as tf
import h5py

from hat import __config__ as C
from hat.util import get_ex
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
        `.m5`文件: 存放元数据，包括参数和原始权重

    Args:
      None
  """
  def __init__(
      self,
      extra_byte=0,
      maxbyte=None,
      h5_suffix=None,
      m5_suffix=None,
      m5_dir='',
      **kwargs):
    self.extra_byte = extra_byte
    self.maxbyte = maxbyte
    self.h5_suffix = h5_suffix
    self.m5_suffix = m5_suffix
    # inner attributes
    self.model = None
    self.h5_dir = ''
    self.m5 = {}
    self.m5_dir = m5_dir
    self.len = 0
    self.batch_size = 0
    self.batch_index = 0
    self.package = []
    self.package_index = 0
    self.train_loss = []
    self.train_accuracy = []
    del kwargs

  # public method

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
    # init Record
    self.init()
    # set learning phase
    if learning_phase is not None:
      nn.set_learning_phase(learning_phase)
    # init_m5
    self.init_m5()

  def on_train_end(self, learning_phase=None):
    """Called at the end of training.
    
      Args:
        learning_phase: Int. Value must be `1` or `0`.
    """
    # set learning phase
    if learning_phase is not None:
      nn.set_learning_phase(learning_phase)
    # end_m5

  def on_batch_end(self, step, x, result, **kwargs):
    outputs = {}
    # step
    outputs['step'] = step
    if config.get('is_write_middle_data'):
      # middle outputs
      mid_output_layers = nn.get_layer_output_name_full(self.model)
      for name in mid_output_layers:
        outputs[name + '_output'] = nn.get_layer_output(
            self.model, x, name).astype(C.get('his_dtype'))
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
    # write h5
    self.write_h5(outputs)

  # built-in method

  def init(self):
    log.info(f"Loading Record", name=__name__)
    self.model = self.model or config.get('model').model
    self.batch_size = config.get('batch_size')
    self.maxbyte = self.maxbyte or C.get('maxbyte')
    self.h5_suffix = self.h5_suffix or C.get('suffix')['his']
    self.h5_dir = self.h5_dir or config.get('save_dir')
    self.h5_dir = os.path.join(self.h5_dir, C.get('his_dir'))
    self.m5_suffix = self.m5_suffix or C.get('suffix')['m5']
    self.m5_dir = self.m5_dir or config.get('save_dir')
    self.m5_dir = os.path.join(self.m5_dir, self.m5_suffix)
    if not os.path.exists(self.h5_dir):
      os.makedirs(self.h5_dir)

  def init_m5(self):
    ########################
    # configs
    ########################
    configs = {
        'lib': config.get('lib_name'),
        'dataset': config.get('dataset_name'),
        'model': config.get('model_name'),
        'batch size': config.get('batch_size'),
        'epochs': config.get('step') and 0 or config.get('epochs'),
        'step': config.get('step'),
        'step per log': config.get('step_per_log'),
        'step per val': config.get('step_per_val'),
        'opt': config.get('opt'),
        'loss': config.get('loss'),
        'metrics': config.get('metrics'),
        'dtype': config.get('dtype'),
        'write middle data': config.get('is_write_middle_data'),
        'write loss': config.get('is_write_loss'),
        'run mode': config.get('run_mode'),
        'addition': config.get('addition'),
        'lr alt': config.get('lr_alt'),
        'xgpu': config.get('xgpu'),
        'xgpu number': config.get('xgpu_num'),
        'dir': config.get('save_dir'),}
    if not isinstance(configs['opt'], str):
      configs['opt'] = '_'.join([
          configs['opt']._name,
          get_ex(configs['opt']._hyper['learning_rate'])])
    self.m5 = {**self.m5, **configs}
    ########################
    # predict batch bytes
    ########################
    K = tf.keras.backend
    input_shape = (1,) + K.int_shape(self.model.input)[1:]
    x = K.zeros(input_shape)
    took_byte = 0
    # step
    took_byte += sys.getsizeof(0)
    if config.get('is_write_middle_data'):
      # middle outputs
      mid_output_layers = nn.get_layer_output_name_full(self.model)
      for name in mid_output_layers:
        took_byte += sys.getsizeof(nn.get_layer_output(self.model, x,
            name).astype(C.get('his_dtype'))) * self.batch_size
    # middle weights
    mid_weight_layers = nn.get_layer_weight_name(self.model)
    if config.get('is_write_middle_data'):
      for name in mid_weight_layers:
        for w in nn.get_layer_weight(self.model, name):
          took_byte += sys.getsizeof(w)
    # metrics
    took_byte += sys.getsizeof(float(0)) * 2
    # extra info
    took_byte += self.extra_byte
    # fix gzip
    self.len = self.maxbyte // (took_byte / 4 * 3)
    self.m5['package length'] = self.len
    log.debug(f'package bytes: {took_byte}', name=__name__)
    # init weights
    init_weights = {}
    for name in mid_weight_layers:
      init_weights[name + '_weight'] = nn.get_layer_weight(
          self.model, name)
    self.m5['init weight'] = init_weights
    # log.debug(self.m5, name=__name__)
    ########################
    # write m5
    ########################
    dts = h5py.special_dtype(vlen=str)
    with h5py.File(self.m5_dir, 'a') as hf:
      batch = self.m5
      for item in batch:
        if isinstance(batch[item], dict):
          # weights: dict of list of ndarray
          sub = hf.create_group(item)
          sbatch = batch[item]
          for sitem in sbatch:
            ssub = sub.create_group(sitem)
            for inx, data in enumerate(sbatch[sitem]):
              ssub.create_dataset(str(inx), data=data,
                  compression='gzip', compression_opts=5)
        elif isinstance(batch[item], list):
          # list: metrics
          sub = hf.create_group(item)
          for inx, data in enumerate(batch[item]):
              sub.create_dataset(str(inx), data=data, dtype=dts)
        elif isinstance(batch[item], str):
          hf.create_dataset(item, data=batch[item], dtype=dts)
        else:
          hf.create_dataset(item, data=np.atleast_1d(batch[item]))

  def write_h5(self, batch:dict):
    package_index = int(self.batch_index // self.len)
    h5_name = str(package_index) + self.h5_suffix
    with h5py.File(os.path.join(self.h5_dir, h5_name), 'a') as hf:
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
    self.batch_index += 1


# test
if __name__ == "__main__":
  from hat.app.standard import mlp
  log.init('./unpush/', detail=True)
  log.info('test')
  config.test((28, 28, 1), (10,))
  mod = mlp.mlp()
  config.set('model', mod)
  config.set('save_dir', './unpush/test')
  r = Record()
  r.on_train_begin()

