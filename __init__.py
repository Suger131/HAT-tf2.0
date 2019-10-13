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

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import sys as _sys

_hat_dir = _os.path.dirname(_os.path.abspath(__file__))
if _hat_dir not in _sys.path:
  _sys.path.append(_hat_dir)

del _os, _sys, _absolute_import, _division, _print_function

# from hat import datasets
# from hat import models
# from hat import utils
