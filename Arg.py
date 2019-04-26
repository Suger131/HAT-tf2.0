import os
import time

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import *
from tensorflow.python.framework.config import (set_gpu_per_process_memory_fraction,
                                                set_gpu_per_process_memory_growth)
# 设置显存使用上限50%，按需申请
# set_gpu_per_process_memory_fraction(0.5)
set_gpu_per_process_memory_growth(True)
# 无用代码，作用是让tensorflow打印完信息再输入指令
SGD()

from datasets import *
from models import *


class Args:

  def __init__(self):

    self.START_TIME = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    self.IS_TRAIN = True
    self.IS_TEST = True
    self.IS_SAVE = True
    self.IS_GIMAGE = False
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
    self.IN_ARGS = input('=>').split(' ')
    self.user()

    self.input_processing()
    self.envs_processing()
    self.gargs()
    self.logs()

  def _check_args(self, item, lists, args_name, data='', isdigit=False):
    '''check'''
    if item not in lists:
      return False
    elif self.__dict__[args_name]:
      print(f'[logs] [Error] {args_name} more than one.')
      os._exit(0)
    else:
      self.__dict__[args_name] = isdigit and (int(data) or int(item)) or data or item
      return True

  def _get_args(self, dicts, cover=True):
    """Built-in method."""
    if cover:
      self.__dict__ = {**self.__dict__, **dicts}
    else:
      for i in dicts: self.__dict__[i] = self.__dict__[i] or dicts[i]

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
            self._check_args(i, ['train-only', 'train-o', 'train'], 'RUN_MODE', data='train') or
            self._check_args(i, ['test-only', 'test-o', 'test'], 'RUN_MODE', data='test')): pass

      else:
        self.WARNING_ARGS.append(i)

  def envs_processing(self):

    # mode envs
    if self.RUN_MODE == 'gimage':
      self.IS_TRAIN = False
      self.IS_TEST = False
      self.IS_SAVE = False
      self.IS_GIMAGE = True

    elif self.RUN_MODE == 'train':
      self.IS_TEST = False

    elif self.RUN_MODE == 'test':
      self.IS_TRAIN = False
      self.IS_SAVE = False

    # load user datasets & models (don't cover)
    self.DATASETS_NAME = self.DATASETS_NAME or self.USER_DICT_N['DATASETS_NAME']
    self.MODELS_NAME = self.MODELS_NAME or self.USER_DICT_N['MODELS_NAME']

    # dir args
    self.SAVE_DIR = f'logs\{self.DATASETS_NAME}_{self.MODELS_NAME}'
    self.MODEL_IMG_DIR = 'model_img'
    self.H5_NAME = f'{self.SAVE_DIR}\{self.DATASETS_NAME}_{self.MODELS_NAME}'
    
    # make dir
    self.DIR_LIST = [self.SAVE_DIR, self.MODEL_IMG_DIR]
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

  def gargs(self):

    # get dataset object
    try:
      self.DATASET = eval(self.DATASETS_NAME + '()')
    except:
      raise NameError(f"No such dataset '{self.DATASETS_NAME}' in Datasets.")
    self._get_args(self.DATASET.ginfo()) 

    # get model object
    try:
      self.MODEL = eval(self.MODELS_NAME + '(self.IMAGE_SHAPE, self.NUM_CLASSES)')
    except:
      raise NameError(f"No such model '{self.MODELS_NAME}' in Models.")
    self._get_args(self.MODEL.ginfo(), cover=False)
    
    # check save
    self.MODEL.model = self.SAVE_EXIST and load_model(self.LOAD_NAME) or self.MODEL.model

    # load user args (don't cover)
    for i in self.USER_DICT:
      try:
        self.__dict__[i] = self.__dict__[i] or self.USER_DICT[i]
      except:
        self.__dict__[i] = self.USER_DICT[i]

    # compile model
    self.MODEL.model.compile(optimizer=self.OPT,
                             loss=self.LOSS_MODE,
                             metrics=['accuracy'])

  def logs(self):

    logs = []
    logs.append(f'#' * 32)
    logs.append(f'[logs] {self.START_TIME}')
    for i in self.WARNING_ARGS:
      logs.append(f"[logs] [Warning] '{i}' is not a supported option")
    if self.RUN_MODE in ['', 'train', 'test']:
      logs.append(f'[logs] Datasets: {self.DATASETS_NAME}')
    logs.append(f'[logs] Models: {self.MODELS_NAME}')
    if self.RUN_MODE in ['', 'train']:
      logs.append(f'[logs] Epochs: {self.EPOCHS}')
    if self.RUN_MODE in ['', 'train', 'test']:
      logs.append(f'[logs] Batch_size: {self.BATCH_SIZE}')
    if self.RUN_MODE in ['', 'train']:
      logs.append(f'[logs] Model Optimizer exist' if self.OPT_EXIST else f'[logs] Using Optimizer: {self.OPT}')
    if self.RUN_MODE in ['', 'train']:
      logs.append(f'[logs] h5 exist: {self.LOAD_NAME}' if self.SAVE_EXIST else f'[logs] h5 not exist, create one')
    if self.RUN_MODE in ['test']:
      logs.append(f'[logs] h5 exist: {self.LOAD_NAME}' if self.SAVE_EXIST else f'[logs] [Warning] h5 not exist, testing a fresh model')
    if self.RUN_MODE in ['', 'train']:
      logs.append(f'[logs] logs dir: {self.LOG_DIR}\\')
    if self.RUN_MODE in ['gimage']:
      logs.append(f'[logs] Get model image')
    elif self.RUN_MODE in ['train']:
      logs.append(f'[logs] train only')
    elif self.RUN_MODE in ['test']:
      logs.append(f'[logs] test only')

    for i in logs: print(i)

  def train(self):
    
    if not self.IS_TRAIN: return

    tensorboard_callback = TensorBoard(log_dir=self.LOG_DIR,
                                       histogram_freq=1,
                                       update_freq='batch',
                                       write_graph=True,
                                       write_images=True)

    self.MODEL.model.fit(self.DATASET.train_images,
                         self.DATASET.train_labels,
                         epochs=self.EPOCHS,
                         batch_size=self.BATCH_SIZE,
                         validation_data=(self.DATASET.test_images, self.DATASET.test_labels), 
                         callbacks=[tensorboard_callback])

  def test(self):

    if not self.IS_TEST: return

    self.RESULT = self.MODEL.model.evaluate(self.DATASET.test_images, self.DATASET.test_labels)

    print('[logs] total loss: %.4f, accuracy: %.4f' % tuple(self.RESULT))

  def save(self):

    if not self.IS_SAVE: return

    self.MODEL.model.save(self.SAVE_NAME, include_optimizer=not self.OPT_EXIST)

    print(f'[logs] Successfully save model: {self.SAVE_NAME}')

  def gimage(self):

    if not self.IS_GIMAGE: return

    from tensorflow.python.keras.utils import plot_model

    plot_model(self.MODEL.model,
               to_file=f'{self.MODEL_IMG_DIR}\{self.MODELS_NAME}_model.png',
               show_shapes=True)

    print(f'[logs] Successfully save model image: {self.MODEL_IMG_DIR}\{self.MODELS_NAME}_model.png')
  
  def user(self):
    '''user train args'''
    self.USER_DICT_N = {'DATASETS_NAME': 'mnist',
                        'MODELS_NAME': 'MLP'}
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
