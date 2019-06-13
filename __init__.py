# Copyright 2019 The HAT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
  HAT
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os as _os

from hat import datasets
from hat import models
from hat import utils

_names_with_underscore = ['__version__', '__git_version__', '__compiler_version__', '__cxx11_abi_flag__', '__monolithic_build__']
__all__ = [_s for _s in dir() if not _s.startswith('_')]
__all__.extend([_s for _s in _names_with_underscore])

_hat_dir = _os.path.dirname(_os.path.abspath(__file__))
if _hat_dir not in __path__:
  __path__.append(_hat_dir)

