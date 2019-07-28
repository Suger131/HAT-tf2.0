# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=wildcard-import
# pylint: disable=attribute-defined-outside-init
# pylint: disable=bare-except
# pylint: disable=unused-argument
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import os

import tensorflow as tf

# tf2 显存管理
# from tensorflow.python.framework.config import (set_gpu_per_process_memory_fraction,
#                                                 set_gpu_per_process_memory_growth)
# 设置显存按需申请
# set_gpu_per_process_memory_growth(True)
# 设置显存使用上限50%
# set_gpu_per_process_memory_fraction(0.5)

# tf1 显存管理
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from hat.utils import *
from hat.datasets import *
from hat.models import *


class Args(object):

  def __init__(self):
    # built-in args
    self._paramc = []
    self._logc = []
    self._specialc = []
    self._warning_list = []
    self._log_list = []
    self._dir_list = []
    # envs args
    self.IS_TRAIN = True
    self.IS_VAL = True
    self.IS_SAVE = True
    self.IS_GIMAGE = True
    self.IS_FLOPS = True
    self.IS_ENHANCE = False
    self.XGPU_MODE = False
    self.XGPU_NUM = 0
    self.XGPU_NMAX = 4
    self.DATASETS_NAME = ''
    self.DATASET = None
    self.MODELS_NAME = ''
    self.MODEL = None
    self.RUN_MODE = ''
    self.MODEL_LIB = ''
    self.BATCH_SIZE = 0
    self.EPOCHS = 0
    self.OPT = None
    self.LOSS_MODE = None
    self.METRICS = []
    self.LIB = None
    self.LIB_NAME = ''
    self.ADDITION = ''
    self.LR_ALT = False
    # build
    self.IN_ARGS = input('=>').split(' ')
    self._Log = None
    self._config = None
    self.user()
    self._input_processing()
    self._envs_processing()
    # built

  # private method

  def _error(self, args, text):
    self._Log(args, _T=text, _A='Error')
    os._exit(1)

  def _check_args(self, item, lists, args_name, data='', force_str=False):
    '''check'''
    if item not in lists:
      return False
    elif self.__dict__[args_name]:
      self._error(args_name, 'More than one:')
    else:
      if type(data) == str and data.isdigit() and not force_str:
        data = int(data)
      self.__dict__[args_name] = data or item
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

      elif '=' in i:
        temp = i.split('=')
        _check_list = [
          [['dat', 'datasets'     ], 'DATASETS_NAME'],
          [['mod', 'models'       ], 'MODELS_NAME'],
          [['ep' , 'epochs'       ], 'EPOCHS'],
          [['bat', 'batch_size'   ], 'BATCH_SIZE'],
          [['mode','runmode'      ], 'RUN_MODE'],
          [['-L' , 'lib'          ], 'MODEL_LIB'],
          [['-X',  'xgpu'         ], 'XGPU_NUM'],
          [['-A', 'add','addition'], 'ADDITION', 'force_str'],
        ]
        _check_box = [
          self._check_args(
            temp[0],
            j[0],
            j[1],
            temp[1],
            force_str=True if 'force_str' in j else False)
          for j in _check_list]
        if not any(_check_box):
          self._warning_list.append(f'Unsupported option: {temp[0]}')
          
      else:
        _check_list = [
          [['-G' , 'gimage'      ], 'RUN_MODE'  , 'gimage'],
          [['-NG', 'no-gimage'   ], 'RUN_MODE'  , 'no-gimage'],
          [['-T' , 'train-only'  ], 'RUN_MODE'  , 'train'],
          [['-V' , 'val-only'    ], 'RUN_MODE'  , 'val'],
          [['-E' , 'data-enhance'], 'IS_ENHANCE', True],
          [['-X' , 'xgpu'        ], 'XGPU_MODE' , True],
          [['-L' , 'lr-alt'      ], 'LR_ALT'    , True],
          [['-NF', 'no-flops'    ], 'IS_FLOPS'  , False],
        ]
        _check_box = [self._check_args(i, *j) for j in _check_list]
        if not any(_check_box):
          self._warning_list.append(f'Unsupported option: {i}')

  def _envs_processing(self):

    # models lib setting
    if not self.MODEL_LIB:
      self.MODEL_LIB = 'S'
    self.LIB = MLib(self.MODEL_LIB)
    self.LIB_NAME = NLib(self.MODEL_LIB)

    # XGPU setting
    if self.XGPU_NUM:
      self.XGPU_MODE = True
      if self.XGPU_NUM > self.XGPU_NMAX:
        self._warning_list.append(f"Max NGPU {self.XGPU_NMAX}, but got {self.XGPU_NUM}. Use the Max NGPU.")
        self.XGPU_NUM = self.XGPU_NMAX
    else:
      self.XGPU_NUM = self.XGPU_NMAX
    self.XGPUINFO = {
      'NGPU': self.XGPU_NUM,
    }

    # load user datasets & models (not cover)
    self._get_args(self.USER_DICT_N)

    # dir args
    self.SAVE_DIR = f'logs/{self.LIB_NAME}/{self.MODELS_NAME}_{self.DATASETS_NAME}'
    if self.XGPU_MODE:
      self.SAVE_DIR += '_xgpu'
    # name addition setting
    if self.ADDITION:
      self.SAVE_DIR += self.ADDITION
    self.H5_NAME = f'{self.SAVE_DIR}/save'

    # make dir
    self._dir_list.extend([self.SAVE_DIR])
    for i in self._dir_list:
      if not os.path.exists(i):
        os.makedirs(i)

    # check save exist
    self.SAVE_TIME = 1
    while True:
      if os.path.exists(f'{self.H5_NAME}_{self.SAVE_TIME}.h5'):
        self.SAVE_TIME += 1
      else:
        if self.SAVE_TIME == 1:
          self.LOAD_NAME = ''
        else:
          self.LOAD_NAME = f'{self.H5_NAME}_{self.SAVE_TIME-1}.h5'
        self.SAVE_NAME = f'{self.H5_NAME}_{self.SAVE_TIME}.h5'
        break
    
    # Set Logger
    self._Log = Log(log_dir=self.SAVE_DIR)
    self._Log(self.SAVE_DIR, _T='Logs dir:')
    self._Log(self._warning_list, _A='Warning')
    self._Log(self._log_list)
    self._timer = Timer(self._Log)

    # get dataset object
    call_dataset = globals().get(self.DATASETS_NAME)
    if callable(call_dataset):
      self._Log(self.DATASETS_NAME, _T='Loading Dataset:')
    else:
      self._error(self.DATASETS_NAME, 'Not in Datasets:')
    self.DATASET = call_dataset()
    _dataset = self.DATASET.ginfo()
    self._get_args(_dataset[0])
    self._paramc.append(_dataset[1])
    self._Log(self.DATASETS_NAME, _T='Loaded Dataset:')

    # get model object
    try:
      call_model = getattr(self.LIB, self.MODELS_NAME)
      self._Log(self.MODELS_NAME, _T='Loading Model:')
    except:
      self._error(self.MODELS_NAME, 'Not in Models:')
    if self.XGPU_MODE:
      self.MODEL = call_model(DATAINFO=self.DATAINFO, XGPUINFO=self.XGPUINFO)
    else:
      self.MODEL = call_model(DATAINFO=self.DATAINFO)
    _model = self.MODEL.ginfo()
    self._get_args(_model[0])
    self._paramc.extend(_model)
    
    # load user args (don't cover)
    self._get_args(self.USER_DICT)

    self.MODEL.build(self.LOAD_NAME)
    
    # compile model
    if self.LOAD_NAME:
      # NOTE: normally, h5 include compile.
      # but in XGPU mode, h5 doesn't include compile
      self._Log(self.LOAD_NAME, _T='Load h5:')
    self.MODEL.compile(
      optimizer=self.OPT,
      loss=self.LOSS_MODE,
      metrics=self.METRICS
    )
    self._Log(self.MODELS_NAME, _T='Loaded Model:')
    self._Log(self.LIB_NAME, _T='Model Lib:')

    # get configer
    self._config = Config(f"{self.SAVE_DIR}/config")
    # processing config
    self._special_config()
    self._paramc.append({
      'BATCH_SIZE': self.BATCH_SIZE,
      'EPOCHS': self.EPOCHS,
      'OPT': self.OPT,
      'LOSS_MODE': self.LOSS_MODE,
      'METRICS': self.METRICS,
    })

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
      self._Log('', _L=['Model Optimizer exist.', f'Using Optimizer: {self.OPT}'], _B=self.OPT != None)
      if self.RUN_MODE == 'val':
        self._Log('', _L=['h5 exist.', 'h5 not exist, valing a fresh model.'], _B=self.LOAD_NAME)
      else:
        self._Log('', _L=['h5 exist.', 'h5 not exist, create one.'], _B=self.LOAD_NAME)
      
    if self.IS_ENHANCE:
      self._Log('Enhance data.')
    if self.XGPU_MODE:
      self._Log('Muti-GPUs.')
    if self.LR_ALT:
      self._Log('Learning Rate Alterable.')

  def _fit(self, *args, **kwargs):
    
    # NOTE: Windows Bug
    # a Windows-specific bug in TensorFlow.
    # The fix is to use the platform-appropriate path separators in log_dir
    # rather than hard-coding forward slashes:
    # NOTE: if use the `histogram_freq`, then raise
    # AttributeError: 'NoneType' object has no attribute 'fetches'
    # unknown bug
    self.LOG_DIR = os.path.join(
      self.SAVE_DIR,
      f'TB_{self.SAVE_TIME}',)
    self._Log(self.LOG_DIR + '\\', _T='logs dir:')
    tensorboard_callback = TensorBoard(log_dir=self.LOG_DIR,
                                       update_freq='batch',
                                       write_graph=False,
                                       write_images=True)
    _history=[]
    
    # Data
    if self.DATASET.trian_x == None:
      if self.IS_ENHANCE:
        self.DATASET.get_generator(self.BATCH_SIZE, self.AUG)
        train = self.DATASET.trian_generator
      else:
        self.DATASET.get_generator(self.BATCH_SIZE)
        train = self.DATASET.trian_generator
    elif self.IS_ENHANCE:
      train = self._datagen()
    
    for i in range(self.EPOCHS):
      if self.LR_ALT:
        lr_rt = self._lr_update(i)
        if lr_rt:
          self._Log(f"LR Update: {lr_rt}")

      self._Log(f"Epoch: {i+1}/{self.EPOCHS} train")
      if self.DATASET.train_x == None or self.IS_ENHANCE:
        _train = self.MODEL.fit_generator(
          train,
          epochs=1,
          callbacks=[tensorboard_callback]
        )
      else:
        _train = self.MODEL.fit(
          self.DATASET.train_x,
          self.DATASET.train_y,
          epochs=1,
          batch_size=self.BATCH_SIZE,
          callbacks=[tensorboard_callback]
        )
      self._Log(f"Epoch: {i+1}/{self.EPOCHS} val")
      if self.DATASET.val_x == None:
        _val = self.MODEL.evaluate_generator(
          self.DATASET.val_generator)
      else:
        _val = self.MODEL.evaluate(
          self.DATASET.val_x,
          self.DATASET.val_y,
          batch_size=self.BATCH_SIZE)
      _history.extend([{f"epoch{i+1}_train_{item}": _train.history[item][0] for item in _train.history},
                       dict(zip([f'epoch{i+1}_val_loss', f'epoch{i+1}_val_accuracy'], _val))])
    return _history

  def _datagen(self):
    """
      Image Data Enhancement
    """
    return self.AUG.flow(
      self.DATASET.train_x, 
      self.DATASET.train_y, 
      batch_size=self.BATCH_SIZE,
      shuffle=True
    )

  def _lr_update(self, i):
    st_list = [0, 100, 150, 200]
    lr_list = [0.1, 0.03, 0.009, 0.0027]
    if any([True for st in st_list if i==st]):
      # stage
      if i < st_list[1]:
        stage = 0
      elif i < st_list[2]:
        stage = 1
      elif i < st_list[3]:
        stage = 2
      else:
        stage = 3
      from tensorflow.python.keras.optimizers import SGD
      opt = SGD(lr=lr_list[stage], momentum=.9, decay=5e-4)
      self.MODEL.compile(
        optimizer=opt,
        loss=self.LOSS_MODE,
        metrics=self.METRICS
      )
      return lr_list[stage]
    else:
      return None

  # public method

  def train(self):
    
    if not self.IS_TRAIN: return

    self._Log(f'{self._GLOBAL_EPOCH-self.EPOCHS}/{self._GLOBAL_EPOCH}', _T='Global Epochs:')

    def _fit(*args, **kwargs):
      # NOTE: Windows Bug
      # a Windows-specific bug in TensorFlow.
      # The fix is to use the platform-appropriate path separators in log_dir
      # rather than hard-coding forward slashes:
      # NOTE: if use the `histogram_freq`, then raise
      # AttributeError: 'NoneType' object has no attribute 'fetches'
      # unknown bug
      self.LOG_DIR = os.path.join(
        self.SAVE_DIR,
        f'TB_{self.SAVE_TIME}',)
      tensorboard_callback = TensorBoard(
        log_dir=self.LOG_DIR,
        update_freq='batch',
        write_graph=False,
        write_images=True
      )
      _history=[]
      
      # Data
      if self.DATASET.train_x is None:
        self.DATASET.get_generator(self.BATCH_SIZE, aug=self.AUG if self.IS_ENHANCE else None)
        train = self.DATASET.trian_generator
      elif self.IS_ENHANCE:
        train = self._datagen()
      
      for i in range(self.EPOCHS):
        if self.LR_ALT:
          lr_rt = self._lr_update(i)
          if lr_rt:
            self._Log(f"LR Update: {lr_rt}")

        self._Log(f"Epoch: {i+1}/{self.EPOCHS} train")
        if self.DATASET.train_x is None or self.IS_ENHANCE:
          _train = self.MODEL.fit_generator(
            train,
            epochs=1,
            callbacks=[tensorboard_callback]
          )
        else:
          _train = self.MODEL.fit(
            self.DATASET.train_x,
            self.DATASET.train_y,
            epochs=1,
            batch_size=self.BATCH_SIZE,
            callbacks=[tensorboard_callback]
          )
        self._Log(f"Epoch: {i+1}/{self.EPOCHS} val")
        if self.DATASET.val_x is None:
          _val = self.MODEL.evaluate_generator(
            self.DATASET.val_generator)
        else:
          _val = self.MODEL.evaluate(
            self.DATASET.val_x,
            self.DATASET.val_y,
            batch_size=self.BATCH_SIZE)
        _history.extend([{f"epoch{i+1}_train_{item}": _train.history[item][0] for item in _train.history},
                        dict(zip([f'epoch{i+1}_val_loss', f'epoch{i+1}_val_accuracy'], _val))])
      return _history

    _, result = self._timer.timer('train', _fit)

    self._logc.extend([_, *result])

  def val(self):

    if not self.IS_VAL: return
    
    def _val():
      if self.DATASET.val_x is None:
        self.DATASET.get_generator(self.BATCH_SIZE)
        _val = self.MODEL.evaluate_generator(
          self.DATASET.val_generator)
      else:
        _val = self.MODEL.evaluate(
          self.DATASET.val_x,
          self.DATASET.val_y,
          batch_size=self.BATCH_SIZE)
      return _val
    
    _, result = self._timer.timer('val', _val)
    self._Log(result[0], _T='total loss:')
    self._Log(result[1], _T='accuracy:')
    self._logc.append(_)
    self._logc.append(dict(zip(['val_total_loss', 'val_accuracy'], result)))

  def test(self):
    return

  def save(self):

    if not self.IS_SAVE: return

    self.MODEL.save(self.SAVE_NAME)

    self._Log(self.SAVE_NAME, _T='Successfully save model:')

  def gimage(self):

    img_name = f'{self.H5_NAME}_model.png'
    ops_name = f'{self.SAVE_DIR}/flops.txt'

    if self.IS_GIMAGE and not os.path.exists(img_name): 

      from tensorflow.python.keras.utils import plot_model

      plot_model(self.MODEL.model,
                to_file=img_name,
                show_shapes=True)

      self._Log(img_name, _T='Successfully save model image:')

    if self.IS_FLOPS and not os.path.exists(ops_name):

      self._paramc.append({
        'FLOPs': self.MODEL.flops(filename=ops_name)
      })

      self._Log(ops_name, _T='Successfully write FLOPs infomation:')
  
  def user(self):
    '''user train args'''
    self.AUG = ImageDataGenerator(
      rotation_range=10,
      width_shift_range=0.05,
      height_shift_range=0.05,
      shear_range=0.05,
      zoom_range=0.05,
      horizontal_flip=True,
    )
    self.USER_DICT_N = {
      'DATASETS_NAME': 'mnist',
      'MODELS_NAME': 'mlp'
    }
    self.USER_DICT = {
      'BATCH_SIZE': 128,
      'EPOCHS': 5,
      'OPT': 'adam',
      'LOSS_MODE': 'sparse_categorical_crossentropy',
      'METRICS': ['accuracy']
    }

  def run(self):

    self.gimage()

    self.train()

    self.val()

    self.test()

    self.save()

    if self.RUN_MODE != 'gimage':
      self._write_config()
    
ARGS = Args()
