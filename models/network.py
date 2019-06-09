from tensorflow.python.keras.models import load_model

class NetWork(object):

  """
  这是一个网络模型基类。

  你需要重写的方法是: 
  
    args()\n\n 定义模型需要的各种参数\n\n 另，可定义BATCH_SIZE，EPOCHS和OPT

    build_model()\n\n 构建网络模型

    build_model里应当包含self.model的定义
  """

  def __init__(self, *args, **kwargs):
    
    self.INPUT_SHAPE = ()
    self.NUM_CLASSES = 0

    self._kwargs = kwargs
    self._default_list = ['BATCH_SIZE', 'EPOCHS', 'OPT', 'OPT_EXIST', 'LOSS_MODE', 'METRICS']
    self._default_dict = {}
    self._dict = {}
    self._check_kwargs()

    self._built = True
    self.BATCH_SIZE = 0
    self.EPOCHS = 0
    self.OPT = None
    self.OPT_EXIST = False
    self.LOSS_MODE = ''
    self.METRICS = []
    self.args()
    self._built = False

    self.build_model()

  # built-in method

  def __setattr__(self, name, value):
    if '_built' not in self.__dict__:
      return super().__setattr__(name, value)
    if name == '_built':
      return super().__setattr__(name, value)
    if self._built:
      if name in self._default_list:
        self._default_dict[name] = value
      else:
        self._dict[name] = value
    return super().__setattr__(name, value)

  # private method

  def _check_kwargs(self):
    if 'DATAINFO' in self._kwargs:
      self.__dict__ = {**self.__dict__, **self._kwargs.pop('DATAINFO')}
    self.__dict__ = {**self.__dict__, **self._kwargs}

  def args(self):
    pass

  def build_model(self):
    raise NotImplementedError

  # public method

  def ginfo(self):
    return self._default_dict, self._dict
