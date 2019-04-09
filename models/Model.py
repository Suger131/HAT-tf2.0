from tensorflow.python.keras.models import load_model

class Model:

  def __init__(self, Args):
    self.ARGS = Args
    self.check_save()

  def check_save(self):
    self.model = self.ARGS.SAVE_EXIST and load_model(self.ARGS.LOAD_NAME) or None

  def ginfo(self):
    for i in self.__dict__:
      if i in ['BATCH_SIZE', 'EPOCHS', 'OPT', 'OPT_EXIST']:
        self.ARGS.__dict__[i] = self.ARGS.__dict__[i] or self.__dict__[i]
