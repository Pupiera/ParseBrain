import os
from parsebrain import dataio
from parsebrain import processing
from parsebrain import utils
from parsebrain import speechbrain_custom
from parsebrain import decoders
from parsebrain import nnet

with open(os.path.join(os.path.dirname(__file__), "version.txt")) as f:
    version = f.read().strip()

__version__ = version
