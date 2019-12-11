# -*- coding: utf-8 -*-
"""HAT
  =====

  Version: 
    3.0 - alpha

  Description: 
    Huster's Artificial neural network Trainer

  Requirements:
    Tensorflow 2.0+ (or 1.14+)(gpu recommended)
    CUDA 10.0.130 (if gpu)
    cuDNN 7.6.5 (if gpu)
    graphviz
    pydot_ng
    Pillow
    gast 0.2.2 (if tf 1.14+)

  License:
    Copyright 2019 The HAT Authors. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import sys as _sys

_hat_dir = _os.path.dirname(_os.path.abspath(__file__))
if _hat_dir not in _sys.path:
  _sys.path.append(_hat_dir)

del _os, _sys, _absolute_import, _division, _print_function

# ============
# Import Part
# ============

from hat.__config__ import version as __version__

from hat import utils
# from hat import util
from hat import dataset
from hat import model

from hat.utils import *
# from hat.util import *
from hat.dataset import *
from hat.model import *
