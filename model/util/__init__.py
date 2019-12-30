# -*- coding: utf-8 -*-
"""__init__

  File: 
    /hat/model/util

  Description: 
    import settings
"""


from hat.model.util import network_v1
from hat.model.util import network_v2
from hat.model.util import nn

from hat.model.util.network_v1 import Network_v1
from hat.model.util.network_v2 import Network_v2


# Alias
# default Network is v2
Network = Network_v2

