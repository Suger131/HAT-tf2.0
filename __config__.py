# -*- coding: utf-8 -*-
"""HAT Config

  File:
    /hat/__config__

  Description:
    HAT config
"""


from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import sys as _sys

hat_dir = _os.path.dirname(_os.path.abspath(__file__))
if hat_dir not in _sys.path:
  _sys.path.append(hat_dir)

del _os, _sys, _absolute_import, _division, _print_function


author = ["York Su"]
github = "https://github.com/YorkSu/hat"
release = False
short_version = '3.0'
full_version = '3.0 - alpha'
version = release and short_version or full_version


__version__ = version
__root__ = hat_dir

