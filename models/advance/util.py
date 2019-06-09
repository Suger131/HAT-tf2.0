# pylint: disable=no-name-in-module
from models.advance.extendrgb import ExtendRGB 
from models.advance.squeezeexcitation import SqueezeExcitation
from models.advance.shuffle import Shuffle
from models.advance.groupconv import GroupConv
from models.advance.swish import Swish
from models.advance.dropconnect import DropConnect
from models.advance.enci import EfficientNetConvInitializer
from models.advance.endi import EfficientNetDenseInitializer

from tensorflow.python.keras.utils.generic_utils import get_custom_objects


# Short name
SE = SqueezeExcitation
ENCI = EfficientNetConvInitializer
ENDI = EfficientNetDenseInitializer


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

