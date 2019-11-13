"""
  hat.model.utils.Network_v2
"""

# pylint: disable=no-name-in-module
# pylint: disable=import-error
# pylint: disable=assignment-from-none


__all__ = [
  'Network_v2'
]


from hat.model.utils import nn


class Network_v2(object):
  """A `Network` is model which is built with `hat.model.utils.nn.layers`

    NOTE: Default `Network` is `Network_v2`
    
    # Built-in:
      nn: hat.model.utils.nn, A powerful module of NN-tools.
      config: hat.utils.config, A powerful parameter management class.

    # Usage
      You need to rewrite `build()` method, define and return a `keras.Model`.
      If you need to define parameters, you can rewrite the `args()` method.

    # Sample:
    ```python
    class lenet(Network):
      def args(self):
        self.node = 500
        self.drop = 0.5
        self.block1 = self.nn.Block('Graph')
        self.block2 = self.nn.Block('Node')
      def build(self):
        inputs = self.nn.input(self.config.input_shape)
        # block1
        x = self.nn.conv(20, 5, padding='valid', activation='relu', block=self.block1)(inputs)
        x = self.nn.maxpool(3, 2, block=self.block1)(x)
        x = self.nn.conv(50, 5, padding='valid', activation='relu', block=self.block1)(x)
        x = self.nn.maxpool(3, 2, block=self.block1)(x)
        # block2
        x = self.nn.flatten(block=self.block2)(x)
        x = self.nn.dense(self.node, activation='relu', block=self.block2)(x)
        x = self.nn.dropout(self.drop, block=self.block2)(x)
        x = self.nn.dense(self.config.output_shape[-1], activation='softmax', block=self.block2)(x)
        return self.nn.model(inputs, x)
    ```
  """
  def __init__(self, 
    config,
    built=True,
    **kwargs):
    self.nn = nn
    self.config = config
    self.config.model = self
    self._pre_built = built
    self.model = None
    self.parallel_model = None
    self.parameter_dict = {
      'batch_size': 0,
      'epochs': 0,
      'opt': None
    }
    # default parameters
    for item in self.parameter_dict:
      self.__dict__[item] = self.parameter_dict[item]
    # dataset parameters
    self.args()
    # feedback to config
    for item in self.parameter_dict:
      self.config.__dict__[item] = self.__dict__[item] or self.config.__dict__[item]
    # pre build
    if self._pre_built:
      self.setup()

  # method for rewrite

  def args(self):
    pass

  def build(self):
    # raise NotImplementedError
    return

  # Built-in method

  def _load(self):
    self.model = self.build()

  def setup(self):
    """
      Setup model
    """
    self._load()
    self.parallel_model = self.model

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
    if not self.config.xgpu:
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
    if self.config.xgpu:
      self.parallel_model.compile(
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
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False,
          **kwargs):
    """
      Get fit function
    """
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
      validation_freq=validation_freq,
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
               callbacks=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False):
    """
      Get evaluate function
    """
    return self.parallel_model.evaluate(
      x=x,
      y=y,
      batch_size=batch_size,
      verbose=verbose,
      sample_weight=sample_weight,
      steps=steps,
      callbacks=callbacks,
      max_queue_size=max_queue_size,
      workers=workers,
      use_multiprocessing=use_multiprocessing
    )

  def predict(self,
              x,
              batch_size=None,
              verbose=1,
              steps=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):
    """
      Get predict function
    """
    return self.parallel_model.predict(
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

  def evaluate_generator(self,
                         generator,
                         steps=None,
                         max_queue_size=10,
                         workers=1,
                         use_multiprocessing=False,
                         verbose=1):
    """
      Get evaluate_generator function
    """
    return self.parallel_model.evaluate_generator(
      generator,
      steps=steps,
      max_queue_size=max_queue_size,
      workers=workers,
      use_multiprocessing=use_multiprocessing,
      verbose=verbose,
    )

  def save(self,
           filepath,
           overwrite=True,
           include_optimizer=True):
    """
      Get save function
    """
    self.model.save(
      filepath,
      overwrite=overwrite,
      include_optimizer=include_optimizer
    )

  def summary(self,
              line_length=100,
              positions=None,
              print_fn=None):
    """
      Get summary function
    """
    self.model.summary(
      line_length=line_length,
      positions=positions,
      print_fn=print_fn
    )

  def flops(self, filename='', hide_re=''):
    # NOTE:
    # tf.RunMetadata -> tf.compat.v1.RunMetadata
    # tf.profiler -> tf.compat.v1.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    if filename:
      opts['output'] = f'file:outfile={filename}'
    ignore = []#['training.*', 'loss.*', 'replica.*']
    if hide_re:
      ignore.append(hide_re)
    opts['hide_name_regexes'] = ignore
    flops = tf.compat.v1.profiler.profile(
      graph=K.get_session().graph,
      run_meta=run_meta,
      cmd='op',
      options=opts
    )
    outputs = flops.total_float_ops
    print(f'FLOPs: {outputs}')
    return outputs


# test
if __name__ == "__main__":
  # from hat.model.utils import nn
  # n = Network_v2()
  # n.nn = nn
  # x_in = n.nn.input((28,28,1))
  # x = n.nn.flatten()(x_in)
  # model = n.nn.model(x_in, x)
  # model.summary()
  from hat.utils._TC import _TC
  t = _TC()
  n = Network_v2(t)
  print(n.model)


