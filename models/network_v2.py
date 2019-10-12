# pylint: disable=unnecessary-pass
# pylint: disable=no-name-in-module

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model

__all__ = [
  'NetWork_v2'
]


class M_Base(Model):
  """
    Base Model of Stem, Body, Head and so on.
    For _M_Main
    For NetWork_v2

    #Argu:
      trainable: bool. It decides whether to train this section.
      name: str. The name of the part of the model.

    #Note:
      You need to rewrite some func. Including:
        self.args: You have to define the self.mtype.
        self.builder: This is the operation part. 
    
    #Sample:
    
    ```python
      def args(self):
        self.mtype = 'Stem'

      def builder(self, inputs):
        x = self.conv(...)(inputs)
        x = self.pool(...)(x)
        return x
    ```
  """
  def __init__(
    self,
    trainable: bool = True,
    name: str = 'Base_Model',
    *args,
    **kwargs
  ):
    self.trainable = trainable
    self.training = None
    self.args()
    if 'mtype' not in self.__dict__:
      raise Exception(f'{self.__class__.__name__}.args() must define self.mtype.')
    name = f'{name}_{self.mtype}'
    super().__init__(name=name, *args, **kwargs)

  def args(self):
    self.mtype = 'Untitled'

  def builder(self, inputs):
    return inputs

  def call(self, inputs, training=True, out_shape=()):
    self.training = True
    if out_shape:
      self.out_shape = out_shape
    return self.builder(inputs)

  def get_layer_outputs(self):
    _L = self.get_config()['layers']
    _L.pop(0)
    _O = [{i['name']: self.get_layer(i['name']).output} for i in _L]
    return _O


class M_Main(Model):
  """
    Main Model for NetWork_v2

    Argu:
      MB_List: List of M_Base. The order of the list is the order in which it runs.
      name: str. The name of the Model.

  """
  def __init__(
      self,
      MB_List: list([M_Base]) = [],
      name: str = 'M_Main',
      *args,
      in_shape=(),
      out_shape=(),
      **kwargs
    ):
    super().__init__(name=name, *args, **kwargs)
    self.MB_List = MB_List
    self.in_shape = in_shape
    self.out_shape = out_shape
    
  def call(self, inputs, training=True):
    x = inputs
    for i in self.MB_List:
      if i.mtype == 'Head':
        x = i(x, training, out_shape=self.out_shape)
      else:
        x = i(x, training)
    return x

  def get_layer_outputs(self):
    _O = []
    # Input
    _input_layer = self.MB_List[0].get_config()['layers'][0]
    _input_name = _input_layer['name']
    _O.append({_input_name: self.MB_List[0].get_layer(_input_name).output})
    # Other
    for i in self.MB_List:
      _O.extend(i.get_layer_outputs())
    return _O


class NetWork_v2(object):
  """
    这是一个网络模型基类 v2
  """
  def __init__(
      self,
      DATAINFO: dict,
      *args, **kwargs
    ):
    self.DATAINFO = DATAINFO
    self.in_shape = DATAINFO['in_shape']
    self.out_shape = DATAINFO['out_shape']
    self.name = 'NetWork_v2'
    self.MB_List = []
    self.model = None

  def args(self):
    pass

  def builder(self) -> list: 
    return []

  def build(self):
    self.args()
    self.MB_List = self.builder()
    inputs = tf.keras.layers.Input(shape=self.in_shape, name="Input")
    outputs = M_Main(self.MB_List, out_shape=self.out_shape)(inputs)
    self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
    
    
# Test
if __name__ == "__main__":
  class mlp(NetWork_v2):
    def args(self):
      name = 'mlp'
    def builder(self):
      class mlp_Head(M_Base):
        def args(self):
          self.mtype = 'Head'
        def builder(self, inputs):
          x = inputs
          x = tf.keras.layers.Flatten()(x)
          x = tf.keras.layers.Dense(128, activation='relu')(x)
          x = tf.keras.layers.Dropout(0.5)(x)
          x = tf.keras.layers.Dense(self.out_shape[0], activation='softmax')(x)
          return x
      return [mlp_Head()]
  M = mlp(DATAINFO={'in_shape': (32, 32, 3), 'out_shape': (10,)})
  M.build()
  M.model.summary()

