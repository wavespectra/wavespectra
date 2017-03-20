import logging
import unittest
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal

import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

# New imports
import xarray as xr
from spectra import SpecArray
from spectra.io import read_ww3, read_swan, read_ww3_msl

D2R=np.pi/180.

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=10)


class TestSpecSwan(unittest.TestCase):
    s=read_swan('prelud.spec')
