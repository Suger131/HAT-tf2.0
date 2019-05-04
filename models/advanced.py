"""
  再次封装的一些功能
  包含的类：
    AdvNet

"""


from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *

from utils.counter import Counter


_COUNT = Counter()


class AdvNet(object):

  def __init__(self, *args, **kwargs):
    return super().__init__(*args, **kwargs)

  def conv(self, x_in, filters, kernel_size, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal'):
    '''卷积层'''
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               kernel_initializer=kernel_initializer,
               name=f"CONV_{_COUNT.get('conv')}_F{filters}_K{type(kernel_size)==int and kernel_size or '%sx%s' % kernel_size}_S{strides}")(x_in)
    return x
  
  def bn(self, x_in, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'):
    '''BN层'''
    x = BatchNormalization(axis=-1,
                           momentum=momentum,
                           epsilon=epsilon,
                           gamma_initializer=gamma_initializer,
                           name='BN_' + str(_COUNT.get('bn')))(x_in)
    return x

  def relu(self, x_in):
    '''RELU层'''
    x = Activation('relu', name='RELU_' + str(_COUNT.get('relu')))(x_in)
    return x

  def conv_bn(self, x_in, filters, kernel_size, strides=1, padding='same', use_bias=False,
              kernel_initializer='he_normal', momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'):
    '''带有bn层的conv'''
    x = self.conv(x_in, filters, kernel_size, strides, padding, use_bias, kernel_initializer)
    x = self.bn(x, momentum, epsilon, gamma_initializer)
    x = self.relu(x)
    return x


class ConvBn(Layer):
  """
    2D convolution layer with BatchNormalization (e.g. spatial convolution over images).

    Arguments:\n
      filters: Integer.\n
      kernel_size: An integer or tuple/list of n integers.\n
      strides: An integer or tuple/list of n integers.\n
      padding: One of `"valid"`,  `"same"`, or `"causal"`.\n
      data_format: A string, one of `channels_last` (default) or `channels_first`.\n
      dilation_rate: An integer or tuple/list of n integers.\n
      activation: Activation function.\n
      use_bias: Boolean.

    Input shape:
      4D tensor with shape:
      `(samples, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, rows, cols, channels)` if data_format='channels_last'.

    Output shape:
      4D tensor with shape:
      `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
  """

  def __init__(self, filters, kernel_size, strides=1, padding='same', activation='relu',
               dilation_rate=1, use_bias=False, kernel_initializer='he_normal', momentum=0.1,
               epsilon=1e-5, gamma_initializer='uniform', axis=-1, data_format=None,**kwargs):
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.activation = activation
    self.dilation_rate = dilation_rate
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.momentum = momentum
    self.epsilon = epsilon
    self.gamma_initializer = gamma_initializer
    self.axis = axis
    self.data_format = conv_utils.normalize_data_format(data_format)
    super(ConvBn, self).__init__(**kwargs)

  def build(self, input_shape):
    super(ConvBn, self).build(input_shape)

  def call(self, x):
    conv = Conv2D(self.filters,
               self.kernel_size,
               strides=self.strides,
               padding=self.padding,
               use_bias=self.use_bias,
               kernel_initializer=self.kernel_initializer)
    bn = BatchNormalization(axis=self.axis,
                           momentum=self.momentum,
                           epsilon=self.epsilon,
                           gamma_initializer=self.gamma_initializer)
    act = Activation(self.activation)
    self.trainable_weights += conv.trainable_weights + bn.trainable_weights + act.trainable_weights
    self.non_trainable_weights += conv.non_trainable_weights + bn.non_trainable_weights + act.non_trainable_weights
    x = conv(x)
    x = bn(x)
    x = act(x)
    return x

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)


if __name__ == "__main__":
  print(_COUNT.get('conv'))
  print(_COUNT.get('conv'))
