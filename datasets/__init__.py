'''
  datasets 子包的init文件
'''

# pylint: disable=wildcard-import

from hat.datasets.utils import *

from hat.datasets.mnist import *
from hat.datasets.cifar import *
from hat.datasets.boston_housing import boston_housing
from hat.datasets.imagenet import imagenet

from hat.datasets.car10 import *
from hat.datasets.dogs import *
from hat.datasets.flower5 import *
from hat.datasets.fruits import *
