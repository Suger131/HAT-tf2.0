# -*- coding: utf-8 -*-
"""Record

  File:
    /hat/core/record

  Description:
    训练过程记录
"""


import gzip
import os
import pickle

import tensorflow as tf

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
    self.package = []
    self.package_index = 0
    self.train_loss = []
    self.train_accuracy = []
    self.init_meta()
    del kwargs

  def update(self, step, x, result, **kwargs):
    outputs = {}
    # step
    outputs['step'] = step
    # middle outputs
    mid_output_layers = nn.get_layer_output_name_full(self.model)
    for name in mid_output_layers:
      outputs[name + '_output'] = nn.get_layer_output(
          self.model, x, name)
    # middle weights
    mid_weight_layers = nn.get_layer_weight_name(self.model)
    for name in mid_weight_layers:
      outputs[name + '_weight'] = nn.get_layer_weight(
          self.model, name)
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

  def get_summary(self):
    outputs = [
        sum(self.train_loss) / len(self.train_loss),
        sum(self.train_accuracy) / len(self.train_accuracy)]
    self.train_loss, self.train_accuracy = [], []
    return outputs

  def init_meta(self):
    # get middle size
    took_byte = 0
    K = tf.keras.backend
    input_shape = (1,) + K.int_shape(self.model.input)[1:]
    x = K.zeros(input_shape)
    layers_name = nn.get_layer_output_name_full(self.model)
    weight_name = nn.get_layer_weight_name(self.model)
    # middle output
    outputs = []
    for name in layers_name:
      outputs.append(nn.get_layer_output(self.model, x, name))
    took_byte += sum([util.quadrature_list(i.shape) for i in outputs]) * \
        self.batch_size
    # weight
    outputs = []
    for name in weight_name:
      outputs.append(nn.get_layer_weight(self.model, name))
    took_byte += sum([util.quadrature_list(j.shape) for i in outputs for j in i])
    # metrics, included loss and accuracy
    took_byte += 2
    # extra info
    took_byte += self.extra_byte
    self.len = self.maxbyte // took_byte
    log.info(f'package max steps: {self.len}', name=__name__)
    # init weights
    init_weights = {}
    for name in weight_name:
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

