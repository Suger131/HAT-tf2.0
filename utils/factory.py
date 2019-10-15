"""
  hat.utils.factory
"""

# pylint: disable=no-name-in-module
# pylint: disable=import-error
# pylint: disable=no-member


__all__ = [
  'factory'
]


import os
from tensorflow.python.keras.callbacks import TensorBoard


class factory(object):
  """
    factory
  """
  def __init__(self, config, *args, **kwargs):
    self.config = config

  # Built-in method

  def _datagen(self):
    """
      Image Data Enhancement
    """
    return self.config.aug.flow(
      self.config.train_x,
      self.config.train_y,
      batch_size=self.config.batch_size,
      shuffle=True
    )

  def _fit(self, *args, **kwargs):
    self.config.log(self.config.tb_dir + '\\', t='Logs dir:')
    tb_callback = TensorBoard(
      log_dir=self.config.tb_dir,
      update_freq='batch',
      write_graph=False,
      write_images=True
    )
    _history = []

    ## data
    if self.config.train_x is None:
      # NOTE: NTF
      # data generater
      if self.config.is_enhance:
        train = self._datagen()
      else:
        train = self._datagen()
    elif self.config.is_enhance:
      train = self._datagen()
  
    ## train
    for i in range(self.config.epochs):
      # lr-alt
      
      # log: epoch train start
      self.config.log(f"Epoch: {i+1}/{self.config.epochs} train")

      # pattern match
      if self.config.train_x is None or self.config.is_enhance:
        _train = self.config.model.fit_generator(
          train,
          epochs=1,
          callbacks=[tb_callback]
        )
      else:
        _train = self.config.model.fit(
          self.config.train_x,
          self.config.train_y,
          epochs=1,
          batch_size=self.config.batch_size,
          callbacks=[tb_callback]
        )
      
      # log: epoch val start
      self.config.log(f"Epochs: {i+1}/{self.config.epochs} val")

      # pattern match
      if self.config.val_x is None:
        _val = self.config.model.evaluate_generator(
          self.config.val_generator
        )
      else:
        _val = self.config.model.evaluate(
          self.config.val_x,
          self.config.val_y,
          batch_size=self.config.batch_size
        )

      # collect history
      _history.extend([{f"epoch{i+1}_train_{item}": _train.history[item][0] for item in _train.history},
                      dict(zip([f'epoch{i+1}_val_loss', f'epoch{i+1}_val_accuracy'], _val))])
      return _history

  def _val(self, *args, **kwargs):
    if self.config.val_x is None:
      self.config.get_generator(self.config.batch_size)
      _val = self.config.model.evaluate_generator(
        self.config.val_generator
      )
    else:
      _val = self.config.model.evaluate(
        self.config.val_x,
        self.config.val_y,
        batch_size=self.config.batch_size
      )
    return _val

  # Public method

  def train(self):

    if not self.config.is_train: return

    # NOTE: NTF
    # log: global epochs
    self.config.log(f'{0}/{self.config.epochs}', t='Global Epochs:')

    _, result = self.config.timer.timer('train', self._fit)

    self.config._logc.extend([_, *result])

  def val(self):

    if not self.config.is_val: return

    _, result = self.config.timer.timer('val', self._val)
    self.config.log(result[0], t='Total loss:')
    self.config.log(result[1], t='Accuracy:')
    self.config._logc.append(_)
    self.config._logc.append(dict(zip(['val_total_loss', 'val_accuracy'], result)))

  def test(self):

    # NOTE: NTF
    # test part

    return

  def save(self):

    if not self.config.is_save: return

    self.config.model.save(self.config.save_name)

    # log: save dir

    self.config.log(self.config.save_name, t='Successfully saved model:')

  def gimage(self):

    img_name = f'{self.config.h5_name}.png'

    if not self.config.is_gimage or os.path.exists(img_name): return
      
    from tensorflow.python.keras.utils import plot_model

    plot_model(
      self.config.model.model,
      to_file=img_name,
      show_shapes=True
    )

    self.config.log(img_name, t='Successfully saved model image:')

  def flops(self):
    
    ops_name = f'{self.config.save_name}/flops.txt'

    if not self.config.is_flops or os.path.exists(ops_name): return

    # NOTE: NTF
    # flops part

    self.config.log(ops_name, t='Successfully wrote FLOPs infomation:')

  def run(self):
    
    self.gimage()
    self.flops()
    self.train()
    self.val()
    self.test()
    self.save()

    # Write config file
    # if self.config.run_mode != 'gimage':
    #   self.write_config()


