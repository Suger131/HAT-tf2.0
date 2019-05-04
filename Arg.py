import os
import time

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


from Log import Log
from datasets import *
from models import *


class Args:

  def __init__(self):

    self.START_TIME = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    self.IS_TRAIN = True
    self.IS_TEST = True
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
    self.WARNING_ARGS = []
    self.DIR_LIST = []
    self.IN_ARGS = input('=>').split(' ')
    self.user()
    self.Log = Log(log_dir='logs/logger')

    self.input_processing()
    self.envs_processing()

  @property
  def _time(self):
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

  def _strptime(self, timex):
    return time.mktime(time.strptime(timex, '%Y-%m-%d-%H-%M-%S'))

  def _error(self, args, text):
    self.Log(args, _T=text, _A='Error')
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

  def input_processing(self):

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
          self.WARNING_ARGS.append(temp[0])

      elif (self._check_args(i, ['gimage', 'gimg'], 'RUN_MODE', data='gimage') or
            self._check_args(i, ['no-gimage', 'n-gimg'], 'RUN_MODE', data='no-gimage') or
            self._check_args(i, ['train-only', 'train-o', 'train'], 'RUN_MODE', data='train') or
            self._check_args(i, ['test-only', 'test-o', 'test'], 'RUN_MODE', data='test')): pass

      else:
        self.WARNING_ARGS.append(i)
    self.Log(self.WARNING_ARGS, _A='Warning', text='Unsupported option:')

  def envs_processing(self):

    # load user datasets & models (don't cover)
    self._get_args(self.USER_DICT_N)

    # dir args
    self.SAVE_DIR = f'logs\{self.DATASETS_NAME}_{self.MODELS_NAME}'
    self.H5_NAME = f'{self.SAVE_DIR}\{self.DATASETS_NAME}_{self.MODELS_NAME}'
    
    # make dir
    self.DIR_LIST.extend([self.SAVE_DIR])
    for i in self.DIR_LIST:
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
    callable(call_dataset) and self.Log(self.DATASETS_NAME, _T='Loading Dataset:') or self._error(self.DATASETS_NAME, 'Not in Datasets:')
    self.DATASET = call_dataset()
    self._get_args(self.DATASET.ginfo())
    self.Log(self.DATASETS_NAME, _T='Loaded Dataset:')

    # get model object
    call_model = globals().get(self.MODELS_NAME)
    callable(call_model) and self.Log(self.MODELS_NAME, _T='Loading Model:') or self._error(self.MODELS_NAME, 'Not in Models:')
    self.MODEL = call_model(DATAINFO=self.DATAINFO)
    self._get_args(self.MODEL.ginfo())
    self.Log(self.MODELS_NAME, _T='Loaded Model:')

    # load user args (don't cover)
    self._get_args(self.USER_DICT)

    # mode envs
    if self.RUN_MODE == 'no-gimage':
      self.IS_GIMAGE = False
      self.Log('Not get model image.')
    if self.RUN_MODE == 'gimage':
      self.IS_TRAIN = False
      self.IS_TEST = False
      self.IS_SAVE = False
      self.Log('Get model image only.')
    elif self.RUN_MODE == 'train':
      self.IS_TEST = False
      self.Log('train only.')
    elif self.RUN_MODE == 'test':
      self.IS_TRAIN = False
      self.IS_SAVE = False
      self.Log('test only.')

    # log some mode info
    if self.RUN_MODE not in ['gimage']:
      self.Log(self.EPOCHS, _T='Epochs:')
      self.Log(self.BATCH_SIZE, _T='Batch size:')
      self.Log('', _L=['Model Optimizer exist.', f'Using Optimizer: {self.OPT}'], _B=self.OPT_EXIST)
      if self.RUN_MODE == 'test':
        self.Log('', _L=['h5 exist.', 'h5 not exist, testing a fresh model.'], _B=self.SAVE_EXIST)
      else:
        self.Log('', _L=['h5 exist.', 'h5 not exist, create one.'], _B=self.SAVE_EXIST)
      self.Log(self.LOG_DIR + '\\', _T='logs dir:')

    # check save
    self.SAVE_EXIST and self.Log(self.LOAD_NAME ,_T='Loading h5:')
    self.MODEL.model = self.SAVE_EXIST and load_model(self.LOAD_NAME) or self.MODEL.model
    self.SAVE_EXIST and self.Log(self.LOAD_NAME ,_T='Loaded h5:')

    # compile model
    self.MODEL.model.compile(optimizer=self.OPT,
                             loss=self.LOSS_MODE,
                             metrics=['accuracy'])

  def train(self):
    
    if not self.IS_TRAIN: return

    tensorboard_callback = TensorBoard(log_dir=self.LOG_DIR,
                                       histogram_freq=1,
                                       update_freq='batch',
                                       write_graph=True,
                                       write_images=True)

    self.START_TIME = self._time

    self.Log(self.START_TIME, _T='Start training:')

    self.MODEL.model.fit(self.DATASET.train_images,
                         self.DATASET.train_labels,
                         epochs=self.EPOCHS,
                         batch_size=self.BATCH_SIZE,
                         validation_data=(self.DATASET.test_images, self.DATASET.test_labels), 
                         callbacks=[tensorboard_callback])

    self.STOP_TIME = self._time
    self.Log(self.STOP_TIME, _T='Stop training:')
    self.TRAIN_COST_TIME = self._strptime(self.STOP_TIME) - self._strptime(self.START_TIME)
    self.Log(self.TRAIN_COST_TIME, _T='Train time (second):')

  def test(self):

    if not self.IS_TEST: return

    self.START_TIME = self._time
    self.Log(self.START_TIME, _T='Start testing:')

    self.RESULT = self.MODEL.model.evaluate(self.DATASET.test_images, self.DATASET.test_labels)

    self.STOP_TIME = self._time
    self.Log(self.STOP_TIME, _T='Stop testing:')
    self.TEST_COST_TIME = self._strptime(self.STOP_TIME) - self._strptime(self.START_TIME)
    self.Log(self.TEST_COST_TIME, _T='Test time (second):')
    self.Log(self.RESULT[0], _T='total loss:')
    self.Log(self.RESULT[1], _T='accuracy:')

  def save(self):

    if not self.IS_SAVE: return

    self.MODEL.model.save(self.SAVE_NAME, include_optimizer=not self.OPT_EXIST)

    self.Log(self.SAVE_NAME, _T='Successfully save model:')

  def gimage(self):

    if not self.IS_GIMAGE: return

    if os.path.exists(f'{self.H5_NAME}_model.png'): return

    from tensorflow.python.keras.utils import plot_model

    plot_model(self.MODEL.model,
               to_file=f'{self.H5_NAME}_model.png',
               show_shapes=True)

    self.Log(f'{self.H5_NAME}_model.png', _T='Successfully save model image:')
  
  def user(self):
    '''user train args'''
    self.USER_DICT_N = {'DATASETS_NAME': 'mnist',
                        'MODELS_NAME': 'mlp'}
    self.USER_DICT = {'BATCH_SIZE': 128,
                      'EPOCHS': 5,
                      'OPT': 'adam',
                      'LOSS_MODE': 'sparse_categorical_crossentropy'}

  def run(self):

    self.gimage()

    self.train()

    self.test()

    self.save()

ARGS = Args()
