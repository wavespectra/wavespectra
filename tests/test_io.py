import logging
import unittest
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal

import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from spectra.io import read_ww3, read_swan, read_ww3_msl

D2R=np.pi/180.

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=10)

SWAN_TEST_FILES=['prelud.spec','antf0.20170207_06z.bnd.swn'] #Not yet quite right for hot files,'antf0.20170208_06z.hot-001',]
TMP_FILE='/tmp/test.spec'

class TestSwanIO(unittest.TestCase):
    def test_swan_files(self):
        for swanfile in SWAN_TEST_FILES:
            self._test_swan(swanfile)
    
    def _test_swan(self,testfile):
        print ("Reading %s") %testfile
        self.s=read_swan(testfile)
        self.s.to_swan(TMP_FILE)
        print ("Writing %s") %TMP_FILE
        tmpspec=read_swan(TMP_FILE)
        assert_array_almost_equal(self.s.dset['efth'],tmpspec.dset['efth'],decimal=4)
    
if __name__ == '__main__':
    unittest.main()