from tensorflow.python.keras.models import load_model

class NetWork(object):

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
    pass

  # public method

  def ginfo(self):
    return {i: self.__dict__[i] for i in ['BATCH_SIZE', 'EPOCHS', 'OPT', 'OPT_EXIST']}
