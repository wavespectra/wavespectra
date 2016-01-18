
import sys
sys.path.append('../')

import logging
import unittest
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
# New imports
import xray
from spectra import SpecArray

reload(sys)
sys.setdefaultencoding('utf-8')

D2R=np.pi/180.

testfile = '/home/tdurrant/Documents/projects/shell/code/old/shell_spec_test.nc'
dset = xray.open_dataset(testfile)
coords={'direction': 'dir', 'frequency':'freq'}
dset.efth *= D2R
dset.direction += 180
dset.direction %= 360
spec = SpecArray(data_array=dset.efth[:,0],dim_map=coords)
spec.tp()
