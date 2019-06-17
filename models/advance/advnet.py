"""
  再次封装的一些功能
  包含的类：
    AdvNet

"""


# pylint: disable=no-name-in-module
# pylint: disable=wildcard-import
# pylint: disable=useless-super-delegation
# pylint: disable=unused-variable

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *

from hat.utils.counter import Counter
from hat.models.network import NetWork
from hat.models.advance.util import *


# import setting
__all__ = [
  'AdvNet'
]


class AdvNet(NetWork):
  """
    Advanced NetWork Builder

    这是一个网络模型基类(高级)

    你需要重写的方法有:
      args 模型的各种参数，在此定义的所有量都会被写入config里面。另，可定义BATCH_SIZE, EPOCHS, OPT
      build_model 构建网络模型，应该包含self.model的定义
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def args(self):
    super().args()

  def build_model(self):
    super().build_model()

  def repeat(self, func, times, *args, **kwargs):
    """
      Return a python function

      Usage:
      ```python
        x = self.repeat(self.local, 3, 128)(x)
      ```
    """
    
    def _func(x):
      for i in range(times):
        x = func(x, *args, **kwargs)
      return x

    return _func

  def input(self, shape, batch_size=None, dtype=None, sparse=False, tensor=None,
            **kwargs):
    """
      Input Layer
    """
    return Input(
      shape=shape,
      batch_size=batch_size,
      dtype=dtype,
      sparse=sparse,
      tensor=tensor,
      name=f"Input",
      **kwargs
    )

  def reshape(self, x, target_shape, **kwargs):
    """
      Reshape Layer
    """
    x = Reshape(
      target_shape=target_shape,
      name=f"Reshape_{Counter('reshape')}",
      **kwargs
    )(x)
    return x

  def add(self, x, **kwargs):
    """
      Add Layer

      x must be a list
    """
    x = Add(
      name=f"Add_{Counter('add')}",
      **kwargs
    )(x)
    return x

  def concat(self, x, axis=-1, **kwargs):
    """
      Concatenate Layer

      x must be a list
    """
    x = Concatenate(
      axis=axis,
      name=f"Concat_{Counter('concat')}",
      **kwargs
    )(x)
    return x

  def local(self, x, units, activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
    """
      Full Connect Layer
    """
    name = f"Softmax" if activation=='softmax' else f"Local_{Counter('local')}"
    x = Dense(
      units=units,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      name=name,
      **kwargs
    )(x)
    return x

  def dropout(self, x, rate, noise_shape=None, seed=None, **kwargs):
    """
      Dropout Layer
    """
    x = Dropout(
      rate=rate,
      noise_shape=noise_shape,
      seed=seed,
      name=f"Dropout_{Counter('dropout')}",
      **kwargs
    )(x)
    return x

  def flatten(self, x, data_format=None, **kwargs):
    """
      Flatten Layer
    """
    x = Flatten(
      data_format=data_format,
      name=f"Faltten_{Counter('flatten')}",
      **kwargs
    )(x)
    return x

  def maxpool(self, x, pool_size=(2, 2), strides=None, padding='same',
              data_format=None, **kwargs):
    """
      Max Pooling 2D Layer
    """
    if not strides:
      strides = pool_size
    x = MaxPool2D(
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      name=f"MaxPool_{Counter('maxpool')}",
      **kwargs
    )(x)
    return x

  def avgpool(self, x, pool_size=(2, 2), strides=(2, 2), padding='same',
              data_format=None, **kwargs):
    """
      Avg Pooling 2D Layer
    """
    x = AvgPool2D(
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      name=f"AvgPool_{Counter('avgpool')}",
      **kwargs
    )(x)
    return x

  def GAPool(self, x, data_format=None, **kwargs):
    """
      Global Avg Pooling 2D
    """
    x = GlobalAvgPool2D(
      data_format=data_format,
      name=f"GlobalAvgPool_{Counter('gapool')}",
      **kwargs
    )(x)
    return x

  def GMPool(self, x, data_format=None, **kwargs):
    """
      Global Max Pooling 2D
    """
    x = GlobalMaxPool2D(
      data_format=data_format,
      name=f"GlobalMaxPool_{Counter('gmpool')}",
      **kwargs
    )(x)
    return x

  def conv(self, x, filters, kernel_size, strides=(1, 1), padding='same',
           data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None, name='', **kwargs):
    """
      Conv2D Layer
    """
    if type(kernel_size) == int:
      kernel_size = (kernel_size,) * 2
    if type(strides) == int:
      strides = (strides,) * 2
    if not name:
      name = ''.join([
        f"Conv_{Counter('conv')}",
        f"_F{filters}",
        f"_K{'%sx%s' % kernel_size}",
        f"_S{'%sx%s' % strides}"
      ])
    x = Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      name=name,
      **kwargs
    )(x)
    return x

  def conv3d(self, x, filters, kernel_size, strides=(1, 1, 1), padding='same',
             data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True,
             kernel_initializer='glorot_uniform', bias_initializer='zeros',
             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
             kernel_constraint=None, bias_constraint=None, name='', **kwargs):
    """
      Conv3D Layer
    """
    if type(kernel_size) == int:
      kernel_size = (kernel_size,) * 3
    if type(strides) == int:
      strides = (strides,) * 3
    if not name:
      name = ''.join([
        f"Conv3D_{Counter('conv')}",
        f"_F{filters}",
        f"_K{'%sx%sx%s' % kernel_size}",
        f"_S{'%sx%sx%s' % strides}"
      ])
    x = Conv3D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      name=name,
      **kwargs
    )(x)
    return x

  def dwconv(self, x, kernel_size, strides=(1, 1), padding='same', depth_multiplier=1,
             data_format=None, activation=None, use_bias=True,
             depthwise_initializer='glorot_uniform', bias_initializer='zeros',
             depthwise_regularizer=None, bias_regularizer=None,
             activity_regularizer=None, depthwise_constraint=None,
             bias_constraint=None, name='', **kwargs):
    """
      DepthwiseConv2D Layer
    """
    if type(kernel_size) == int:
      kernel_size = (kernel_size,) * 2
    if type(strides) == int:
      strides = (strides,) * 2
    if not name:
      name = ''.join([
        f"DWConv_{Counter('dwconv')}",
        f"_K{'%sx%s' % kernel_size}",
        f"_S{'%sx%s' % strides}"
      ])
    x = DepthwiseConv2D(
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      depth_multiplier=depth_multiplier,
      data_format=data_format,
      activation=activation,
      use_bias=use_bias,
      depthwise_initializer=depthwise_initializer,
      bias_initializer=bias_initializer,
      depthwise_regularizer=depthwise_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      depthwise_constraint=depthwise_constraint,
      bias_constraint=bias_constraint,
      name=name,
      **kwargs
    )(x)
    return x

  def groupconv(self, x, groups, filters, kernel_size, strides=(1, 1), padding='same',
           data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None, split=True, rank=2, name='', **kwargs):
    """
      Group Conv Layer
    """
    if type(kernel_size) == int:
      kernel_size = (kernel_size,) * 2
    if type(strides) == int:
      strides = (strides,) * 2
    if not name:
      name = ''.join([
        f"GroupConv_{Counter('conv')}",
        f"_G{groups}",
        f"_F{filters}",
        f"_K{'%sx%s' % kernel_size}"
      ])
    x = GroupConv(
      groups=groups,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      split=split,
      rank=rank,
      name=name,
      **kwargs
    )(x)
    return x

  def bn(self, x, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
         beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
         moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
         beta_constraint=None, gamma_constraint=None, renorm=False, renorm_clipping=None,
         renorm_momentum=0.99, fused=None, trainable=True, virtual_batch_size=None,
         adjustment=None, **kwargs):
    """
      BatchNormalization Layer
    """
    x = BatchNormalization(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=center,
      scale=scale,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      moving_mean_initializer=moving_mean_initializer,
      moving_variance_initializer=moving_variance_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      beta_constraint=beta_constraint,
      gamma_constraint=gamma_constraint,
      renorm=renorm,
      renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum,
      fused=fused,
      trainable=trainable,
      virtual_batch_size=virtual_batch_size,
      adjustment=adjustment,
      name=f"BN_{Counter('bn')}",
      **kwargs
    )(x)
    return x

  def relu(self, x, **kwargs):
    """
      ReLU Layer
    """
    x = Activation(
      'relu',
      name=f"ReLU_{Counter('relu')}",
      **kwargs
    )(x)
    return x

  def activation(self, x, activation, **kwargs):
    """
      Activation Layer
    """
    x = Activation(
      activation=activation,
      name=f"{activation.capitalize()}_{Counter('relu')}",
      **kwargs
    )(x)
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
      Conv2D with BN and ReLU(optional)
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
                fused=fused, trainable=trainable, virtual_batch_size=virtual_batch_size,
                adjustment=adjustment, **kwargs)
    if activation == 'relu':
      x = self.relu(x, **kwargs)
    else:
      x = self.activation(x, activation, **kwargs)
    return x

  def exrgb(self, x, k, dilation_rate=1, data_format=None, **kwargs):
    """
      拓展RGB通道
    """
    x = ExtendRGB(
      k,
      dilation_rate=dilation_rate,
      data_format=data_format, 
      name=f"Extend_RGB",
      **kwargs
    )(x)
    return x

  def SE(self, x, input_filters=None, rate=16, activation='sigmoid', data_format=None,
        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, **kwargs):
    """
      SqueezeExcitation Layer

      If input_filters=None, input_filters will use input channels(depend on axis)

      Usage:
      ```python
        x = self.SE(x)
        x = self.SE(x, rate=4)
        x = self.SE(x, 128, 4)
      ```
    """
    x = SE(
      input_filters=input_filters,
      rate=rate,
      activation=activation,
      data_format=data_format,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      name=f"SE_{Counter('se')}",
      **kwargs
    )(x)
    return x

  def shuffle(self, x, axis=-1, **kwargs):
    """
      Layer that shuffle and concatenate a list of inputs

      Usage: The same as keras.layers.Concatenate
    """
    x = Shuffle(
      axis=axis,
      name=f"Shuffle_{Counter('shuffle')}",
      **kwargs
    )
    return x

  def swish(self, x, **kwargs):
    """
      Swish Layer(Activation)

      Swish = x * Sigmoid(x)
    """
    x = Swish(
      name=f"Swish_{Counter('swish')}",
      **kwargs
    )(x)
    return x

  def dropconnect(self, x, drop_connect_rate=0, **kwargs):
    """
      Drop Connect Layer
    """
    x = DropConnect(
      drop_connect_rate,
      name=f"DropConnect_{Counter('dropconnect')}",
      **kwargs
    )(x)
    return x

  def proc_input(self, x, size, axis=-1):
    """
      Processing the input tensor shape.

      ### Input:
        x: Input Tensor

        size: Target shape, int or 2D tuple or 2D list
    """
    if isinstance(size, int):
      size = (size, size)
    elif len(size) != 2:
      raise Exception(f'len of size must be 2 if tuple or list, but got: {len(size)}')
    
    if axis==-1:
      _size = K.int_shape(x)[1:3]
    else:
      _size = K.int_shape(x)[2:4]
    
    _n = [(size[i] - _size[i]) // 2 for i in range(2)]
    
    if _n[0] < 0 and _n[1] < 0:
      _n = [-i for i in _n]
      x = Cropping2D(_n)(x)
    elif _n[0] > 0 and _n[1] > 0:
      x = ZeroPadding2D(_n)(x)
    else:
      pass
    
    return x


if __name__ == "__main__":
  print(Counter('conv'))
  print(f"{Counter('conv')}")
