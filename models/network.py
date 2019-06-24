# pylint: disable=unnecessary-pass
# pylint: disable=no-name-in-module

import tensorflow as tf
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.models import load_model

__all__ = [
  'NetWork'
]


class NetWork(object):
  """
    这是一个网络模型基类

    你需要重写的方法有:
      args 模型的各种参数，在此定义的所有量都会被写入config里面。另，可定义BATCH_SIZE, EPOCHS, OPT
      build_model 构建网络模型，应该包含self.model的定义
  """

  def __init__(self, **kwargs):
    
    # DATAINFO
    self.INPUT_SHAPE = ()
    self.NUM_CLASSES = 0
    # XGPUINFO
    self.XGPU = False
    self.NGPU = 0
    
    self.LOAD = False
    self.model = None

    self._kwargs = kwargs
    self._default_list = ['BATCH_SIZE', 'EPOCHS', 'OPT', 'LOSS_MODE', 'METRICS']
    self._default_dict = {}
    self._dict = {}
    self._check_kwargs()

    # catch argument
    self._built = True
    self.BATCH_SIZE = 0
    self.EPOCHS = 0
    self.OPT = None
    self.LOSS_MODE = ''
    self.METRICS = []
    self.args()
    self._built = False

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
    if 'XGPUINFO' in self._kwargs:
      self.XGPU = True
      self.__dict__ = {**self.__dict__, **self._kwargs.pop('XGPUINFO')}
    self.__dict__ = {**self.__dict__, **self._kwargs}

  def args(self):
    """
      定义需要写入到config的参数

      另，可以定义BATCH_SIZE, EPOCHS, OPT
    """
    pass

  def build_model(self):
    """
      构建网络模型，应该包含self.model的定义
    """
    raise NotImplementedError

  # public method

  def ginfo(self):
    return self._default_dict, self._dict

  def build(self, filepath=''):
    """
      Build model

      Argument:
        filepath: Str. If not '', use this file path to load model.
    """
    if filepath:
      self.LOAD = True
    if self.XGPU:
      with tf.device('/cpu:0'):
        if filepath:
          self.model = load_model(filepath)
        else:
          self.build_model()
      try:
        self.parallel_model = multi_gpu_model(self.model, gpus=self.NGPU)
      except:
        print(f'\n[WARNING] XGPU Failed. Check out the Numbers of GPU(NGPU), got {self.NGPU} \n')
        self.XGPU = False
    else:
      if filepath:
        self.model = load_model(filepath)
      else:
        self.build_model()

  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              distribute=None,
              **kwargs):
    """
      Get compile function
    """
    self.model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics,
      loss_weights=loss_weights,
      sample_weight_mode=sample_weight_mode,
      weighted_metrics=weighted_metrics,
      target_tensors=target_tensors,
      distribute=distribute,
      **kwargs
    )
    # if self.XGPU:
    #   self.parallel_model.compile(
    #     optimizer=optimizer,
    #     loss=loss,
    #     metrics=metrics,
    #     loss_weights=loss_weights,
    #     sample_weight_mode=sample_weight_mode,
    #     weighted_metrics=weighted_metrics,
    #     target_tensors=target_tensors,
    #     distribute=distribute,
    #     **kwargs
    #   )

  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False,
          **kwargs):
    """
      Get fit function
    """
    if self.XGPU:
      return self.parallel_model.fit(
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_split=validation_split,
        validation_data=validation_data,
        shuffle=shuffle,
        class_weight=class_weight,
        sample_weight=sample_weight,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        **kwargs
      )
    else:
      return self.model.fit(
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_split=validation_split,
        validation_data=validation_data,
        shuffle=shuffle,
        class_weight=class_weight,
        sample_weight=sample_weight,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        **kwargs
      )

  def evaluate(self,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               sample_weight=None,
               steps=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False):
    """
      Get evaluate function
    """
    if self.XGPU:
      return self.parallel_model.evaluate(
        x=x,
        y=y,
        batch_size=batch_size,
        verbose=verbose,
        sample_weight=sample_weight,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing
      )
    else:
      return self.model.evaluate(
        x=x,
        y=y,
        batch_size=batch_size,
        verbose=verbose,
        sample_weight=sample_weight,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing
      )

  def predict(self,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):
    """
      Get predict function
    """
    if self.XGPU:
      return self.parallel_model.predict(
        x,
        batch_size=batch_size,
        verbose=verbose,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing
      )
    else:
      return self.model.predict(
        x,
        batch_size=batch_size,
        verbose=verbose,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing
      )

  def fit_generator(self,
                    generator,
                    steps_per_epoch=None,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_data=None,
                    validation_steps=None,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0):
    """
      Get fit_generator function
    """
    if self.XGPU:
      return self.parallel_model.fit_generator(
        generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_steps,
        class_weight=class_weight,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        shuffle=shuffle,
        initial_epoch=initial_epoch
      )
    else:
      return self.model.fit_generator(
        generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_steps,
        class_weight=class_weight,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        shuffle=shuffle,
        initial_epoch=initial_epoch
      )

  def save(self,
           filepath,
           overwrite=True,
           include_optimizer=True):
    """
      Get save function
    """
    self.model.save(filepath, overwrite=overwrite, include_optimizer=include_optimizer)

  def summary(self,
              line_length=100,
              positions=None,
              print_fn=None):
    """
      Get summary function
    """
    self.model.summary(line_length=line_length, positions=positions, print_fn=print_fn)
