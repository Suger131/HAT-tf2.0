# -*- coding: utf-8 -*-
"""MLP

  File: 
    /hat/model/test/mlp

  Description: 
    MLP模型
    简单三层神经网络[添加了Dropout层]
    *基于Network_v2
"""


import hat


# import setting
__all__ = [
    'mlp',]


class mlp(hat.Network):
  """MLP

    Description:
      MLP模型，简单三层神经网络
      添加了Dropout

    Args:
      None

    Overrided:
      args: 存放参数
      build: 定义了`keras.Model`并返回
  """
  def args(self):
    self.node = 128
    self.drop = 0.5

  def build(self):
    inputs = self.nn.input(self.input_shape)
    x = self.nn.flatten()(inputs)
    x = self.nn.dense(self.node)(x)
    x = self.nn.dropout(self.drop)(x)
    x = self.nn.dense(self.output_class, activation='softmax')(x)
    return self.nn.model(inputs, x)

import gzip
import os
import pickle
import tensorflow as tf
from hat import __config__ as C
from hat.util import util
from hat.model.util import nn


class _Record(object):
  """Record"""
  def __init__(
      self,
      model,
      batch_size,
      save_dir,
      extra_byte=0,
      maxbyte=C.get('maxbyte'),
      suffix='.gz',
      **kwargs):
    self.model = model
    self.batch_size = batch_size
    self.dir = save_dir
    self.extra_byte = extra_byte
    self.maxbyte = maxbyte
    self.suffix = suffix

    self.len = maxbyte // self.get_size()
    self.package = []
    self.package_index = 0
    
    self.train_loss = []
    self.train_accuracy = []
    del kwargs

  def get_size(self):
    K = tf.keras.backend
    input_shape = K.int_shape(self.model.input)[1:]
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
    with gzip.open(os.path.join(
        self.dir,
        str(self.package_index) + self.suffix), 'wb') as f:
      pickle.dump(self.package, f)
    self.package = []
    self.package_index += 1


# test
if __name__ == "__main__":
  t = hat.util.Tc()
  t.data.input_shape = (28, 28, 1)
  t.data.output_shape = (10,)
  mod = mlp(config=t)
  mod.summary()
  r = _Record(mod.model, 128, './')
  print(r.get_size())

