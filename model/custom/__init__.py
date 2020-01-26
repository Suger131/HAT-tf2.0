# -*- coding: utf-8 -*-
"""__init__

  File: 
    /hat/model/custom

  Description: 
    import settings
"""


from tensorflow.keras.utils import get_custom_objects

from hat.model.custom import initializer
from hat.model.custom import layer
from hat.model.custom import optimizer


CUSTOM_OBJECTS = {
    'Swish': layer.Swish,
    'HSigmoid': layer.HSigmoid,
    'HSwish': layer.HSwish,
    'AddBias': layer.AddBias,
    'ResolutionScal2D': layer.ResolutionScal2D,
    'ResolutionScaling2D': layer.ResolutionScaling2D,
    'GroupConv2D': layer.GroupConv2D,
    'LoopDense': layer.LoopDense,
    'LoopConv2D': layer.LoopConv2D,
    'SqueezeExcitation': layer.SqueezeExcitation,
    'SE': layer.SE,}
get_custom_objects().update(CUSTOM_OBJECTS)

