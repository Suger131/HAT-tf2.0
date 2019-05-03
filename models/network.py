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
    self.kwargs = kwargs
    self._check_kwargs()
    self.BATCH_SIZE = 0
    self.EPOCHS = 0
    self.OPT = None
    self.OPT_EXIST = False
    self.args()
    self.build_model()

  # private method

  def _check_kwargs(self):
    if 'DATAINFO' in self.kwargs:
      self.__dict__ = {**self.__dict__, **self.kwargs['DATAINFO']}
      self.kwargs.pop('DATAINFO')
    self.__dict__ = {**self.__dict__, **self.kwargs}

  def args(self):
    pass

  def build_model(self):
    raise NotImplementedError

  # public method

  def ginfo(self):
    return {i: self.__dict__[i] for i in ['BATCH_SIZE', 'EPOCHS', 'OPT', 'OPT_EXIST']}
