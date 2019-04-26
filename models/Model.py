from tensorflow.python.keras.models import load_model

class BasicModel:

  def __init__(self):
    self.BATCH_SIZE = 0
    self.EPOCHS = 0
    self.OPT = None
    self.OPT_EXIST = False

  # def check_save(self):
  #   if self.ARGS:
  #     self.model = self.ARGS.SAVE_EXIST and load_model(self.ARGS.LOAD_NAME) or self.model

  # def ginfo(self):
  #   for i in self.__dict__:
  #     if i in ['BATCH_SIZE', 'EPOCHS', 'OPT', 'OPT_EXIST']:
  #       self.ARGS.__dict__[i] = self.ARGS.__dict__[i] or self.__dict__[i]

  def ginfo(self):
    return {i: self.__dict__[i] for i in ['BATCH_SIZE', 'EPOCHS', 'OPT', 'OPT_EXIST']}
