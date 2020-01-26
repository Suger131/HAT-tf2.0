# -*- coding: utf-8 -*-
"""__init__

  File: 
    /hat/model/custom/layer

  Description: 
    import settings
"""


from hat.model.custom.layer import activation
from hat.model.custom.layer import basic
from hat.model.custom.layer import graph
from hat.model.custom.layer import group
from hat.model.custom.layer import loop
from hat.model.custom.layer import squeeze


from hat.model.custom.layer.activation import Swish
from hat.model.custom.layer.activation import HSigmoid
from hat.model.custom.layer.activation import HSwish

from hat.model.custom.layer.basic import AddBias

from hat.model.custom.layer.graph import ResolutionScal2D

from hat.model.custom.layer.group import GroupConv2D

from hat.model.custom.layer.loop import LoopDense
from hat.model.custom.layer.loop import LoopConv2D

from hat.model.custom.layer.squeeze import SqueezeExcitation


# Alias
ResolutionScaling2D = ResolutionScal2D
SE = SqueezeExcitation

