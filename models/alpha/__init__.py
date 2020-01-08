"""
  Alpha Models

"""

# conv standard simple network
from hat.models.alpha.cnn32 import cnn32
from hat.models.alpha.cnn64 import cnn64
from hat.models.alpha.cnn128 import cnn128
from hat.models.alpha.cnn256 import cnn256

# experiment network
from hat.models.alpha.mlp import mlp
from hat.models.alpha.dwdnet import dwdnet
from hat.models.alpha.nvgg16 import nvgg16
from hat.models.alpha.wrn28_10 import wrn28_10
from hat.models.alpha.stagenet import stagenet
from hat.models.alpha.lenet_3d import lenet_3d
from hat.models.alpha.shufflenet import shufflenet
from hat.models.alpha.enet import enet
from hat.models.alpha.enb0_ci import enb0_ci
from hat.models.alpha.lsg import lsg
from hat.models.alpha.lg import lg
from hat.models.alpha.vgg10 import vgg10
