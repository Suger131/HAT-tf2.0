
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops


class Shuffle(_Merge):
  """
    Layer that shuffle and concatenate a list of inputs

    Usege: The same as keras.layers.Concatenate
  """
  def __init__(self, axis=-1, **kwargs):
    super(Shuffle, self).__init__(**kwargs)
    self.axis = axis
    self.supports_masking = True
    self._reshape_required = False

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # Used purely for shape validation.
    if not isinstance(input_shape, list) or len(input_shape) < 2:
      raise ValueError('A `Concatenate` layer should be called '
                       'on a list of at least 2 inputs')
    if all(shape is None for shape in input_shape):
      return
    reduced_inputs_shapes = [list(shape) for shape in input_shape]
    shape_set = set()
    for i in range(len(reduced_inputs_shapes)):
      del reduced_inputs_shapes[i][self.axis]
      shape_set.add(tuple(reduced_inputs_shapes[i]))
    if len(shape_set) > 1:
      raise ValueError('A `Concatenate` layer requires '
                       'inputs with matching shapes '
                       'except for the concat axis. '
                       'Got inputs shapes: %s' % (input_shape))

  def _merge_function(self, inputs):
    
    x = K.concatenate(inputs, axis=self.axis)
    
    _len = len(inputs)
    _shape = K.int_shape(x)[1:]
    _shapex = _shape[:-1] + (_len, _shape[-1] // _len)
    _transpose = list(range(len(_shapex)+1))
    _transpose = _transpose[:-2] + [_transpose[-1], _transpose[-2]]

    x = Reshape(_shapex)(x)
    x = tf.transpose(x, _transpose)
    x = Reshape(_shape)(x)

    return x 

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if not isinstance(input_shape, list):
      raise ValueError('A `Concatenate` layer should be called '
                       'on a list of inputs.')
    input_shapes = input_shape
    output_shape = list(input_shapes[0])
    for shape in input_shapes[1:]:
      if output_shape[self.axis] is None or shape[self.axis] is None:
        output_shape[self.axis] = None
        break
      output_shape[self.axis] += shape[self.axis]
    return tuple(output_shape)

  def compute_mask(self, inputs, mask=None):
    if mask is None:
      return None
    if not isinstance(mask, list):
      raise ValueError('`mask` should be a list.')
    if not isinstance(inputs, list):
      raise ValueError('`inputs` should be a list.')
    if len(mask) != len(inputs):
      raise ValueError('The lists `inputs` and `mask` '
                       'should have the same length.')
    if all(m is None for m in mask):
      return None
    # Make a list of masks while making sure
    # the dimensionality of each mask
    # is the same as the corresponding input.
    masks = []
    for input_i, mask_i in zip(inputs, mask):
      if mask_i is None:
        # Input is unmasked. Append all 1s to masks,
        masks.append(array_ops.ones_like(input_i, dtype='bool'))
      elif K.ndim(mask_i) < K.ndim(input_i):
        # Mask is smaller than the input, expand it
        masks.append(array_ops.expand_dims(mask_i, axis=-1))
      else:
        masks.append(mask_i)
    concatenated = K.concatenate(masks, axis=self.axis)
    return K.all(concatenated, axis=-1, keepdims=False)

  def get_config(self):
    config = {
        'axis': self.axis,
    }
    base_config = super(Shuffle, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
