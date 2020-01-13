# -*- coding: utf-8 -*-
"""Network

  File: 
    /hat/model/network

  Description: 
    Network 基类
"""


import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.utils import multi_gpu_model
# from tensorflow.keras.models import load_model

from hat.core import abc
from hat.core import log
from hat.model import nn


class Network_v1(abc.Keras_Network):
  """Network V1

    Description: 
      A base class of Network Model.
      You need to rewrite `build()` method, define and return 
      a `keras.Model`(or nn.model). If you need to define 
      parameters, you can rewrite the `args()` method.

    Args:
      config: hat.Config.
      built: Bool, default True.

    Attributes:
      nn: hat.model.util.nn, A module of NNs based on keras.
      input_shape: hat.config.data.input_shape
      output_shape: hat.config.data.output_shape
      output_class: classes of dataset, if `classfication`
  """
  def __init__(
      self,
      config,
      built=True,
      **kwargs):
    self.config = config
    self._pre_built = built
    self.model = None
    self.parallel_model = None
    self.xgpu = self.config.xgpu
    self.parameter_dict = {
        'batch_size': 0,
        'epochs': 0,
        'opt': None}
    # default parameters
    for item in self.parameter_dict:
      self.__dict__[item] = self.parameter_dict[item]
    # dataset parameters
    self.args()
    # feedback to config
    for item in self.parameter_dict:
      self.config.__dict__[item] = self.__dict__[item] or self.config.__dict__[item]
    # pre build
    if self._pre_built:
      self.setup()
    del kwargs

  # Built-in method
  def _load(self):
    self.model = self.build()

  def args(self):
    pass

  def build(self):
    raise NotImplementedError
    # return

  def setup(self):
    """Setup model"""
    self._load()
    self.parallel_model = self.model


class Network_v2(abc.Keras_Network):
  """Network V2

    Description: 
      Default Network
      Build Network with hat.model.util.nn.layers
      You need to rewrite `build()` method, define and return 
      a `keras.Model`(or nn.model). If you need to define 
      parameters, you can rewrite the `args()` method.

    Args:
      config: hat.Config.
      built: Bool, default True.

    Attributes:
      nn: hat.model.util.nn, A module of NNs based on keras.
      input_shape: hat.config.data.input_shape
      output_shape: hat.config.data.output_shape
      output_class: classes of dataset, if `classfication`

    Usage:
    ```python
    class lenet(Network):
      def args(self):
        self.node = 500
        self.drop = 0.5
        self.block1 = self.nn.Block('Graph')
        self.block2 = self.nn.Block('Node')
      def build(self):
        inputs = self.nn.input(self.input_shape)
        # block1
        x = self.nn.conv(20, 5, padding='valid', activation='relu', block=self.block1)(inputs)
        x = self.nn.maxpool(3, 2, block=self.block1)(x)
        x = self.nn.conv(50, 5, padding='valid', activation='relu', block=self.block1)(x)
        x = self.nn.maxpool(3, 2, block=self.block1)(x)
        # block2
        x = self.nn.flatten(block=self.block2)(x)
        x = self.nn.dense(self.node, activation='relu', block=self.block2)(x)
        x = self.nn.dropout(self.drop, block=self.block2)(x)
        x = self.nn.dense(self.output_class, activation='softmax', block=self.block2)(x)
        return self.nn.model(inputs, x)
    ```
  """
  def __init__(self, 
      config=None,
      built=True,
      **kwargs):
    if not config:
      raise Exception('No Config Object')
    self.nn = nn
    self.config = config
    self._pre_built = built
    self.model = None
    self.parallel_model = None
    self.xgpu = config.xgpu
    self.input_shape = config.data.input_shape
    self.output_shape = config.data.output_shape
    if len(self.output_shape) == 1:
      self.output_class = self.output_shape[0]
    else:
      self.output_class = None
    self.parameter_dict = {
        'batch_size': 0,
        'epochs': 0,
        'opt': None}
    # default parameters
    for item in self.parameter_dict:
      self.__dict__[item] = self.parameter_dict[item]
    # dataset parameters
    self.args()
    # feedback to config
    for item in self.parameter_dict:
      self.config.__dict__[item] = self.__dict__[item] or self.config.__dict__[item]
    # pre build
    if self._pre_built:
      self.setup()

  # method for rewrite

  def args(self):
    pass

  def build(self):
    # raise NotImplementedError
    return

  # Built-in method

  def load(self):
    if self.config.load_name:
      log.info(f'Load {self.config.load_name}',
          name=__name__)
      model = tf.keras.models.load_model(self.config.load_name)
    else:
      model = self.build()
      if not isinstance(model, tf.keras.models.Model):
        log.error(f'build() must return `tf.keras.models.Model`, '\
            f'but got {type(model)}', name=__name__, exit=True)
    return model

  def setup(self):
    """Setup model"""
    self.model = self.load()
    self.parallel_model = self.model


# Alias
Network = Network_v2 # Default Network


# test
if __name__ == "__main__":
  from hat.util import Tc
  t = Tc()
  t.data.input_shape = (28, 28, 1)
  t.data.output_shape = (10,)
  n = Network_v2(t)
  print(n.model)

