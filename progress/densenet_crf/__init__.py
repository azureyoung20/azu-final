import numpy as np

np.random.seed(0)

from wsi.densenet_crf.densenet import (densenet121)


MODELS = {'densenet121': densenet121}