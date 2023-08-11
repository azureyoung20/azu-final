import numpy as np

np.random.seed(0)

from progress.densenet_baseline.densenet import (densenet121)


MODELS = {'densenet121': densenet121}