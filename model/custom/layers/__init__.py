# -*- coding: utf-8 -*-
"""__init__

  File: 
    /hat/model/custom/layers

  Description: 
    import settings
"""


from hat.model.custom.layers import activation
from hat.model.custom.layers import basic
from hat.model.custom.layers import graph
from hat.model.custom.layers import group
from hat.model.custom.layers import squeeze


from hat.model.custom.layers.activation import Swish
from hat.model.custom.layers.activation import HSigmoid
from hat.model.custom.layers.activation import HSwish

from hat.model.custom.layers.basic import AddBias

from hat.model.custom.layers.graph import ResolutionScaling2D

from hat.model.custom.layers.group import GroupConv2D

from hat.model.custom.layers.squeeze import SqueezeExcitation


# Alias
ResolutionScal2D = ResolutionScaling2D
SE = SqueezeExcitation

