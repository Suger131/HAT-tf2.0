# -*- coding: utf-8 -*-
"""Basic Config

  File:
    /hat/util/__config__

  Description:
    config setting
"""


from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator # pylint: disable=import-error


NAME_MAP = {
    # ================
    # set
    # ================

    # dataset_name
    'd': {'n': 'dataset_name', 'force_str': True},
    'dat': {'n': 'dataset_name', 'force_str': True},
    'dataset': {'n': 'dataset_name', 'force_str': True},
    # lib_name
    'l': {'n': 'lib_name', 'force_str': True},
    'lib': {'n': 'lib_name', 'force_str': True},
    # model_name
    'm': {'n': 'model_name', 'force_str': True},
    'mod': {'n': 'model_name', 'force_str': True},
    'model': {'n': 'model_name', 'force_str': True},
    # batch_size
    'b': {'n': 'batch_size', 'l': True},
    'bat': {'n': 'batch_size', 'l': True},
    'batchsize': {'n': 'batch_size', 'l': True},
    # epochs
    'e': {'n': 'epochs', 'l': True},
    'ep': {'n': 'epochs', 'l': True},
    'epochs': {'n': 'epochs', 'l': True},
    # steps
    's': {'n': 'step', 'l': True},
    'step': {'n': 'step', 'l': True},
    # step pre log
    'spl': {'n': 'step_per_log', 'l': True},
    'stepperlog': {'n': 'step_per_log', 'l': True},
    # opt
    'o': {'n': 'opt', 'l': True},
    'opt': {'n': 'opt', 'l': True},
    # run_mode
    'r': {'n': 'run_mode', 'force_str': True},
    'rm': {'n': 'run_mode', 'force_str': True},
    'runmode': {'n': 'run_mode', 'force_str': True},
    # addition
    'a': {'n': 'addition', 'force_str': True},
    'add': {'n': 'addition', 'force_str': True},
    'addition': {'n': 'addition', 'force_str': True},
    # dtype
    't': {'n': 'dtype'},
    'type': {'n': 'dtype'},
    # xgpu
    'x': {'n': 'xgpu'},

    # ================
    # tag
    # ================

    # gimage
    '-g': {'n': 'run_mode', 'd': 'gimage'},
    'gimage': {'n': 'run_mode', 'd': 'gimage'},
    # no-gimage
    '-ng': {'n': 'run_mode', 'd': 'no-gimage'},
    'no-gimage': {'n': 'run_mode', 'd': 'no-gimage'},
    # train-only
    '-t': {'n': 'run_mode', 'd': 'train'},
    'train-only': {'n': 'run_mode', 'd': 'train'},
    # val-only
    '-v': {'n': 'run_mode', 'd': 'val'},
    'val-only': {'n': 'run_mode', 'd': 'val'},
    # data enhance
    '-e': {'n': 'is_enhance', 'd': True},
    'd-e': {'n': 'is_enhance', 'd': True},
    'data-enhance': {'n': 'is_enhance', 'd': True},
    # xgpu
    '-x': {'n': 'xgpu', 'd': -1},
    'xgpu': {'n': 'xgpu', 'd': -1},
    # learning rate alterable
    '-l': {'n': 'lr_alt', 'd': True},
    'lr-alt': {'n': 'lr_alt', 'd': True},
    # no-flops
    '-nf': {'n': 'is_flops', 'd': False},
    'no-flops': {'n': 'is_flops', 'd': False},
    # write middle data
    '-w': {'n': 'is_write_middle_data', 'd': True},
    '-wmd': {'n': 'is_write_middle_data', 'd': True},
    'write-middle-data': {'n': 'is_write_middle_data', 'd': True},
    # no-write-loss
    '-nwl': {'n': 'is_write_loss', 'd': False},
    'no-write-loss': {'n': 'is_write_loss', 'd': False},
    # gpu memory growth
    '-ngg': {'n': 'gpu_growth', 'd': False},
    'no-gpu-growth': {'n': 'gpu_growth', 'd': False},}


AUG = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,)


DEFAULT_SETTING = {
    'dataset_name': 'mnist',
    'lib_name': 'standard',
    'model_name': 'mlp',
    'batch_size': 128,
    'epochs': 5,
    'step': 0,
    'step_per_log': 10,
    'step_per_val': 500,
    'opt': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy'],
    'dtype': 'float32',
    'aug': AUG,
    'is_train': True,
    'is_val': True,
    'is_test': False,
    'is_save': True,
    'is_save_best': False,
    'is_gimage': True,
    'is_flops': False,
    'is_enhance': False,
    'is_write_middle_data': False,
    'is_write_loss': True,
    'run_mode': '',
    'addition': '',
    'lr_alt': False}


INIT_SETTING = {
    'name_map': NAME_MAP,
    'aug': AUG,
    'default': DEFAULT_SETTING,
    'log_root': 'logs',
    'log_name': '',
    'h5_name': 'save',
    'tb_dir': 'tensorboard',
    'gpu_growth': True}


def get(name):
  return INIT_SETTING[name]

