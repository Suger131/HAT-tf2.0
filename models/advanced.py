"""
  再次封装的一些功能
  包含的类：
    AdvNet

"""


from tensorflow.python.keras.layers import *

from utils.counter import Counter


class AdvNet(object):

  def __init__(self, *args, **kwargs):
    return super().__init__(*args, **kwargs)

  def repeat(self, func, times, x, *args, **kwargs):
    for i in range(times):
      x = func(x, *args, **kwargs)
    return x

  def input(self, shape, batch_size=None, dtype=None, sparse=False, tensor=None,
            **kwargs):
    return Input(shape=shape, batch_size=batch_size, dtype=dtype, sparse=sparse,
                 tensor=tensor, name=f"Input", **kwargs)

  def local(self, x, units, activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
    name = f"Softmax" if activation=='softmax' else f"Local_{Counter('local')}"
    x = Dense(units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
              kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
              name=name, **kwargs)(x)
    return x

  def dropout(self, x, rate, noise_shape=None, seed=None, **kwargs):
    x = Dropout(rate=rate, noise_shape=noise_shape, seed=seed,
                name=f"Dropout_{Counter('dropout')}", **kwargs)(x)
    return x

  def flatten(self, x, data_format=None, **kwargs):
    x = Flatten(data_format=data_format, name=f"Faltten_{Counter('flatten')}",
                **kwargs)(x)
    return x

  def maxpool(self, x, pool_size=(2, 2), strides=None, padding='same',
              data_format=None, **kwargs):
    if not strides:
      strides = pool_size
    x = MaxPool2D(pool_size=pool_size, strides=strides, padding=padding,
                  data_format=data_format, name=f"MaxPool_{Counter('maxpool')}", **kwargs)(x)
    return x

  def avgpool(self, x, pool_size=(2, 2), strides=(2, 2), padding='same',
              data_format=None, **kwargs):
    x = AvgPool2D(pool_size=pool_size, strides=strides, padding=padding,
                  data_format=data_format, name=f"AvgPool_{Counter('avgpool')}", **kwargs)(x)
    return x

  def GAPool(self, x, data_format=None, **kwargs):
    x = GlobalAvgPool2D(data_format=data_format, name=f"GlobalAvgPool_{Counter('gapool')}", **kwargs)(x)
    return x

  def GMPool(self, x, data_format=None, **kwargs):
    x = GlobalMaxPool2D(data_format=data_format, name=f"GlobalMaxPool_{Counter('gmpool')}", **kwargs)(x)
    return x

  def conv(self, x, filters, kernel_size, strides=(1, 1), padding='same',
           data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None, **kwargs):
    if type(kernel_size) == int:
      kernel_size = (kernel_size,) * 2
    if type(strides) == int:
      strides = (strides,) * 2
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
               padding=padding, data_format=data_format, dilation_rate=dilation_rate,
               activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
               name=f"Conv_{Counter('conv')}" +
                    f"_F{filters}" +
                    f"_K{'%sx%s' % kernel_size}" +
                    f"_S{'%sx%s' % strides}", **kwargs)(x)
    return x

  def bn(self, x, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
         beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
         moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
         beta_constraint=None, gamma_constraint=None, renorm=False, renorm_clipping=None,
         renorm_momentum=0.99, fused=None, trainable=True, virtual_batch_size=None,
         adjustment=None, **kwargs):
    x = BatchNormalization(axis=axis, momentum=momentum, epsilon=epsilon, center=center,
                           scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                           moving_mean_initializer=moving_mean_initializer, moving_variance_initializer=moving_variance_initializer,
                           beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                           beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
                           renorm=renorm, renorm_clipping=renorm_clipping, renorm_momentum=renorm_momentum,
                           fused=fused, trainable=trainable, virtual_batch_size=virtual_batch_size,
                           adjustment=adjustment, name=f"BN_{Counter('bn')}", **kwargs)(x)
    return x

  def relu(self, x, **kwargs):
    x = Activation('relu', name=f"ReLU_{Counter('relu')}", **kwargs)(x)
    return x

  def activation(self, x, activation, **kwargs):
    x = Activation(activation=activation, name=f"{activation.capitalize()}_{Counter('relu')}",
                   **kwargs)(x)
    return x

  def conv_bn(self, x, filters, kernel_size, strides=(1, 1), padding='same',
           data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None, axis=-1, momentum=0.99, epsilon=1e-3,
           center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
           moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None,
           gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, renorm=False,
           renorm_clipping=None, renorm_momentum=0.99, fused=None, trainable=True,
           virtual_batch_size=None, adjustment=None, **kwargs):
    '''
      带有bn层的conv, 默认激活函数为ReLU
    '''
    x = self.conv(x, filters=filters, kernel_size=kernel_size, strides=strides,
                  padding=padding, data_format=data_format, dilation_rate=dilation_rate,
                  activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                  bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                  kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
    x = self.bn(x, axis=axis, momentum=momentum, epsilon=epsilon, center=center,
                scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                moving_mean_initializer=moving_mean_initializer, moving_variance_initializer=moving_variance_initializer,
                beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
                renorm=renorm, renorm_clipping=renorm_clipping, renorm_momentum=renorm_momentum,
                fused=fused, trainable=trainable, virtual_batch_size=virtual_batch_size, **kwargs)
    if activation == 'relu':
      x = self.relu(x, **kwargs)
    else:
      x = self.activation(x, activation, **kwargs)
    return x


if __name__ == "__main__":
  print(Counter('conv'))
  print(f"{Counter('conv')}")
