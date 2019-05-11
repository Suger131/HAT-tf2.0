import os

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import *
# from tensorflow.python.framework.config import (set_gpu_per_process_memory_fraction,
#                                                 set_gpu_per_process_memory_growth)
# 设置显存使用上限50%，按需申请
# set_gpu_per_process_memory_fraction(0.5)
# set_gpu_per_process_memory_growth(True)
# tf1 显存管理
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


from models.advanced import _CUSTOM_OBJECTS
from utils import *
from datasets import *
from models import *


class Args(object):

  def __init__(self):
    # built-in args
    self._paramc = []
    self._logc = []
    self._specialc = []
    self._warning_args = []
    self._dir_list = []
    self._CUSTOM_OBJECTS = _CUSTOM_OBJECTS
    # envs args
    self.IS_TRAIN = True
    self.IS_VAL = True
    self.IS_SAVE = True
    self.IS_GIMAGE = True
    self.DATASETS_NAME = ''
    self.DATASET = None
    self.MODELS_NAME = ''
    self.MODEL = None
    self.RUN_MODE = ''
    self.BATCH_SIZE = 0
    self.EPOCHS = 0
    self.OPT = None
    self.OPT_EXIST = False
    self.LOSS_MODE = None
    self.METRICS = []
    # build
    self.IN_ARGS = input('=>').split(' ')
    self._Log = Log(log_dir='logs/logger')
    self._timer = Timer(self._Log)
    self._config = None
    self.user()
    self._input_processing()
    self._envs_processing()
    # built

  # private method

  def _error(self, args, text):
    self._Log(args, _T=text, _A='Error')
    os._exit(1)

  def _check_args(self, item, lists, args_name, data='', isdigit=False):
    '''check'''
    if item not in lists:
      return False
    elif self.__dict__[args_name]:
      self._error(args_name, 'More than one:')
    else:
      self.__dict__[args_name] = isdigit and (int(data) or int(item)) or data or item
      return True

  def _get_args(self, dicts, cover=False):
    """Built-in method."""
    if cover:
      self.__dict__ = {**self.__dict__, **dicts}
    else:     
      for i in dicts:
        try:
          self.__dict__[i] = self.__dict__[i] or dicts[i]
        except:
          self.__dict__[i] = dicts[i]

  def _write_config(self):
    if not self._config.if_param():
      for di in self._paramc:
        self._config.param(di)
    for di in self._specialc:
      self._config.param(di)
    for di in self._logc:
      self._config.log(di)

  def _special_config(self):
    self._GLOBAL_EPOCH = self._config.get('param', 'global_epoch')
    if self._GLOBAL_EPOCH:
      self._GLOBAL_EPOCH = int(self._GLOBAL_EPOCH)
      self._GLOBAL_EPOCH += self.EPOCHS
    else:
      self._GLOBAL_EPOCH = self.EPOCHS
    self._specialc.append({'GLOBAL_EPOCH': self._GLOBAL_EPOCH})
    self._logc.append({'EPOCHS': self.EPOCHS, 'BATCH_SIZE': self.BATCH_SIZE})

  def _input_processing(self):

    for i in self.IN_ARGS:

      if i in ['', ' ']:
        pass

      elif i.find('=') != -1:
        temp = i.split('=')
        if (self._check_args(temp[0], ['data', 'dataset', 'datasets', 'dat'], 'DATASETS_NAME', temp[1]) or
            self._check_args(temp[0], ['model', 'models', 'mod'], 'MODELS_NAME', temp[1]) or
            self._check_args(temp[0], ['epochs', 'epoch', 'ep'], 'EPOCHS', temp[1], isdigit=True) or
            self._check_args(temp[0], ['batch_size', 'bat'], 'BATCH_SIZE', temp[1], isdigit=True) or
            self._check_args(temp[0], ['mode'], 'RUN_MODE', temp[1])): pass
        else:
          self._warning_args.append(temp[0])

      elif (self._check_args(i, ['gimage', 'gimg'], 'RUN_MODE', data='gimage') or
            self._check_args(i, ['no-gimage', 'n-gimg'], 'RUN_MODE', data='no-gimage') or
            self._check_args(i, ['train-only', 'train-o', 'train'], 'RUN_MODE', data='train') or
            self._check_args(i, ['val-only', 'val-o', 'val'], 'RUN_MODE', data='val')): pass

      else:
        self._warning_args.append(i)
    self._Log(self._warning_args, _A='Warning', text='Unsupported option:')

  def _envs_processing(self):

    # load user datasets & models (don't cover)
    self._get_args(self.USER_DICT_N)

    # dir args
    self.SAVE_DIR = f'logs\{self.DATASETS_NAME}_{self.MODELS_NAME}'
    self.H5_NAME = f'{self.SAVE_DIR}\{self.DATASETS_NAME}_{self.MODELS_NAME}'
    
    # make dir
    self._dir_list.extend([self.SAVE_DIR])
    for i in self._dir_list:
      if not os.path.exists(i):
        os.makedirs(i)

    # check save exist
    self.SAVE_EXIST = True
    self.SAVE_TIME = 0
    if not os.path.exists(f'{self.H5_NAME}_{self.SAVE_TIME}.h5'):
      self.SAVE_EXIST = False
    else:
      while os.path.exists(f'{self.H5_NAME}_{self.SAVE_TIME}.h5'):
        self.SAVE_TIME += 1
      self.LOAD_NAME = f'{self.H5_NAME}_{self.SAVE_TIME-1}.h5'
    self.SAVE_NAME = f'{self.H5_NAME}_{self.SAVE_TIME}.h5'

    # NOTE: Windows Bug
    # a Windows-specific bug in TensorFlow.
    # The fix is to use the platform-appropriate path separators in log_dir
    # rather than hard-coding forward slashes:
    self.LOG_DIR = os.path.join(f'logs',
                                f'{self.DATASETS_NAME}_{self.MODELS_NAME}',
                                f'{self.DATASETS_NAME}_{self.MODELS_NAME}_{self.SAVE_TIME}',)

    # get dataset object
    call_dataset = globals().get(self.DATASETS_NAME)
    callable(call_dataset) and self._Log(self.DATASETS_NAME, _T='Loading Dataset:') or self._error(self.DATASETS_NAME, 'Not in Datasets:')
    self.DATASET = call_dataset()
    _dataset = self.DATASET.ginfo()
    self._get_args(_dataset[0])
    self._paramc.append(_dataset[1])
    self._Log(self.DATASETS_NAME, _T='Loaded Dataset:')

    # get model object
    call_model = globals().get(self.MODELS_NAME)
    callable(call_model) and self._Log(self.MODELS_NAME, _T='Loading Model:') or self._error(self.MODELS_NAME, 'Not in Models:')
    self.MODEL = call_model(DATAINFO=self.DATAINFO)
    _model = self.MODEL.ginfo()
    self._get_args(_model[0])
    self._paramc.extend(_model)
    self._Log(self.MODELS_NAME, _T='Loaded Model:')

    # load user args (don't cover)
    self._get_args(self.USER_DICT)

    # get configer
    self._config = Config(f"{self.SAVE_DIR}/config")
    # processing special config
    self._special_config()

    # mode envs
    if self.RUN_MODE == 'no-gimage':
      self.IS_GIMAGE = False
      self._Log('Not get model image.')
    if self.RUN_MODE == 'gimage':
      self.IS_TRAIN = False
      self.IS_VAL = False
      self.IS_SAVE = False
      self._Log('Get model image only.')
    elif self.RUN_MODE == 'train':
      self.IS_VAL = False
      self._Log('train only.')
    elif self.RUN_MODE == 'val':
      self.IS_TRAIN = False
      self.IS_SAVE = False
      self._Log('val only.')

    # log some mode info
    if self.RUN_MODE not in ['gimage']:
      self._Log(self.EPOCHS, _T='Epochs:')
      self._Log(self.BATCH_SIZE, _T='Batch size:')
      self._Log('', _L=['Model Optimizer exist.', f'Using Optimizer: {self.OPT}'], _B=self.OPT_EXIST)
      if self.RUN_MODE == 'val':
        self._Log('', _L=['h5 exist.', 'h5 not exist, valing a fresh model.'], _B=self.SAVE_EXIST)
      else:
        self._Log('', _L=['h5 exist.', 'h5 not exist, create one.'], _B=self.SAVE_EXIST)
      self._Log(self.LOG_DIR + '\\', _T='logs dir:')

    # check save
    self.SAVE_EXIST and self._Log(self.LOAD_NAME ,_T='Loading h5:')
    self.MODEL.model = self.SAVE_EXIST and load_model(self.LOAD_NAME, custom_objects=self._CUSTOM_OBJECTS) or self.MODEL.model
    self.SAVE_EXIST and self._Log(self.LOAD_NAME ,_T='Loaded h5:')

    # compile model
    self.MODEL.model.compile(optimizer=self.OPT,
                             loss=self.LOSS_MODE,
                             metrics=self.METRICS)

  def _fit(self, *args, **kwargs):
    # NOTE: if use the `histogram_freq`, then raise
    # AttributeError: 'NoneType' object has no attribute 'fetches'
    # unknown bug
    tensorboard_callback = TensorBoard(log_dir=self.LOG_DIR,
                                      #  histogram_freq=1,
                                       update_freq='batch',
                                       write_graph=False,
                                       write_images=True)
    _history = []
    for i in range(self.EPOCHS):
      self._Log(f"Epoch: {i+1}/{self.EPOCHS} train")
      _train = self.MODEL.model.fit(self.DATASET.train_x,
                                    self.DATASET.train_y,
                                    epochs=1, #self.EPOCHS,
                                    batch_size=self.BATCH_SIZE,
                                    callbacks=[tensorboard_callback])#
      self._Log(f"Epoch: {i+1}/{self.EPOCHS} val")
      _val = self.MODEL.model.evaluate(self.DATASET.val_x,
                                       self.DATASET.val_y,
                                       batch_size=self.BATCH_SIZE)
      _history.extend([{f"epoch{i+1}_train_{item}": _train.history[item][0] for item in _train.history},
                       dict(zip([f'epoch{i+1}_val_loss', f'epoch{i+1}_val_accuracy'], _val))])
    return _history

  # public method

  def train(self):
    
    if not self.IS_TRAIN: return

    self._Log(f'{self._GLOBAL_EPOCH-self.EPOCHS}/{self._GLOBAL_EPOCH}', _T='Global Epochs:')
    _, result = self._timer.timer('train', self._fit)

    self._logc.extend([_, *result])

  def val(self):

    if not self.IS_VAL: return

    _, result = self._timer.timer('val', self.MODEL.model.evaluate,
                                  self.DATASET.val_x,
                                  self.DATASET.val_y,
                                  batch_size=self.BATCH_SIZE)
    self._Log(result[0], _T='total loss:')
    self._Log(result[1], _T='accuracy:')
    self._logc.append(_)
    self._logc.append(dict(zip(['val_total_loss', 'val_accuracy'], result)))

  def test(self):
    return

  def save(self):

    if not self.IS_SAVE: return

    self.MODEL.model.save(self.SAVE_NAME, include_optimizer=not self.OPT_EXIST)

    self._Log(self.SAVE_NAME, _T='Successfully save model:')

  def gimage(self):

    if not self.IS_GIMAGE: return

    if os.path.exists(f'{self.H5_NAME}_model.png'): return

    from tensorflow.python.keras.utils import plot_model

    plot_model(self.MODEL.model,
               to_file=f'{self.H5_NAME}_model.png',
               show_shapes=True)

    self._Log(f'{self.H5_NAME}_model.png', _T='Successfully save model image:')
  
  def user(self):
    '''user train args'''
    self.USER_DICT_N = {'DATASETS_NAME': 'mnist',
                        'MODELS_NAME': 'mlp'}
    self.USER_DICT = {'BATCH_SIZE': 128,
                      'EPOCHS': 5,
                      'OPT': 'adam',
                      'LOSS_MODE': 'sparse_categorical_crossentropy',
                      'METRICS' : ['accuracy']}

  def run(self):

    self.gimage()

    self.train()

    self.val()

    self.test()

    self.save()

    from pprint import pprint

    pprint(self._paramc)
    pprint(self._logc)
    pprint(self._specialc)
    self._write_config()
    
ARGS = Args()
