# -*- coding: utf-8 -*-
"""HAT
  =====

  Version: 
    3.0 - alpha

  Description: 
    Huster's Artificial neural network Trainer

  Requirements:
    Tensorflow 2.0 (or 1.14+)(gpu recommended)
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


# ============
# Import 
# ============


from hat import __config__
__version__ = __config__.__version__
__root__ = __config__.__root__

from hat import core
from hat import dataset
from hat import model
from hat import util

from hat.core import config
from hat.core import factory
from hat.dataset import Dataset
from hat.model import Network
from hat.model import nn

