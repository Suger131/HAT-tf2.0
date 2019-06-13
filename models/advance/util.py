# pylint: disable=no-name-in-module
from tensorflow.python.keras.utils.generic_utils import get_custom_objects

from hat.models.advance.dropconnect import DropConnect
from hat.models.advance.enci import EfficientNetConvInitializer
from hat.models.advance.endi import EfficientNetDenseInitializer
from hat.models.advance.extendrgb import ExtendRGB
from hat.models.advance.groupconv import GroupConv
from hat.models.advance.shuffle import Shuffle
from hat.models.advance.squeezeexcitation import SqueezeExcitation
from hat.models.advance.swish import Swish


# import setting
__all__ = [
  'ExtendRGB',
  'GroupConv',
  'SqueezeExcitation',
  'SE',
  'Shuffle',
  'Swish',
  'DropConnect',
  'EfficientNetConvInitializer',
  'EfficientNetDenseInitializer',
  'ENCI',
  'ENDI',
]


# Short name
SE = SqueezeExcitation
ENCI = EfficientNetConvInitializer
ENDI = EfficientNetDenseInitializer


# envs
_CUSTOM_OBJECTS = {
  'ExtendRGB': ExtendRGB,
  'GroupConv': GroupConv,
  'SqueezeExcitation': SqueezeExcitation,
  'SE': SE,
  'Shuffle': Shuffle,
  'Swish': Swish,
  'DropConnect': DropConnect,
  'EfficientNetConvInitializer': EfficientNetConvInitializer,
  'EfficientNetDenseInitializer': EfficientNetDenseInitializer,
  'ENCI': ENCI,
  'ENDI': ENDI,
}
get_custom_objects().update(_CUSTOM_OBJECTS)
