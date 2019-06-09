# pylint: disable=attribute-defined-outside-init

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils


class ExtendRGB(Layer):
  """
    Extend the RGB channels

    Input:
      (batch, ..., 3)
    
    Output:
      (batch, ..., k*6)

    Usage:

    ```python
      x = ExtendRGB(4)(x) # got (batch, ..., 24)
    ```
  """

  def __init__(self, k, data_format=None, dilation_rate=1, trainable=False, **kwargs):
    super(ExtendRGB, self).__init__(trainable=trainable, **kwargs)
    self.k = k
    self.data_format = data_format
    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
    if self.data_format == 'channels_First':
      self.axis = 1
    else:
      self.axis = -1

  def build(self, input_shape):
    self.kernel = self._color_weight()
    self._convolution_op = K.conv2d
    self.built = True

  def call(self, inputs, **kwargs):
    assert inputs.shape[self.axis] != 3, f"Input Tensor must have 3 channels(RGB), but got {inputs.shape[self.axis]}"
    x = self._convolution_op(
      inputs, 
      self.kernel, 
      padding='same',
      data_format=self.data_format, 
      dilation_rate=self.dilation_rate
    )
    return x

  def _color_weight(self):
    _weight = []
    for i in range(3):
      i_ = i + 1 if i + 1 <= 2 else 0
      for j in range(self.k + 1):
        _t = [0, 0, 0]
        _t[i] = 1. / (1. + j / self.k)
        _t[i_] = j / self.k / (1. + j / self.k)
        _weight.append(_t)
      for j in range(1, self.k):
        _t = [0, 0, 0]
        _t[i_] = 1. / (1. + (self.k - j) / self.k)
        _t[i] = (self.k - j) / self.k / (1. + (self.k - j) / self.k)
        _weight.append(_t)
    _weight = K.variable(_weight)
    _weight = K.transpose(_weight)
    _weight = K.reshape(_weight, (1, 1, 3, 6 * self.k))
    return _weight

  def compute_output_shape(self, input_shape):
    input_shape[self.axis] = 6 * self.k
    return input_shape

  def get_config(self):
    config = {
      'k': self.k,
    }
    base_config = super(ExtendRGB, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
