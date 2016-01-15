import sys
sys.path.append('../')

import logging
import unittest
import time
import matplotlib.pyplot as plt
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')

# New imports
import xray
from spectra import SpecArray

# Old imports
from pymo.data.spectra_new import WW3NCSpecFile,WW3NCGridFile
from pymo.core.basetype import Site

testfile = '/home/tdurrant/Downloads/shell_spec_test.nc'

D2R=np.pi/180.

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=10)


class TestSpec(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print ("%s: %.3f" % (self.id(), t))

    def calcNew(self,method='hs'):
        # New spectra
        startTime = time.time()
        dset = xray.open_dataset(testfile)
        coords={'direction': 'dir', 'frequency':'freq'}
        dset.efth *= D2R
        dset.direction += 180
        dset.direction %= 360
        self.specnew = SpecArray(data_array=dset.efth[:,0],dim_map=coords)
        self.new = getattr(self.specnew,method)()
        print ('%s: pymsl: %s' % (method, time.time() - startTime))

    def calcOld(self,method='hs'):
        # Old spectra
        startTime = time.time()
        self.specold=WW3NCSpecFile(testfile)
        self.sites = [Site(x=x,y=y) for x,y in zip(self.specold.lon,self.specold.lat)]
        self.old = self.calcOldTs(method=method)
        print ('%s: pymo: %s' % (method, time.time() - startTime))

    def test_vars(self):
        #for method in ('hs','tp'):
        for method in ('dp',):
            self.calcOld(method=method)
            self.calcNew(method=method)
            self.plot()
            plt.title(method)
            #np.allclose(self.old.values, self.new.values)
        plt.show()

    def plot(self):
        plt.figure()
        self.old.plot(label='pymo')
        self.new.plot(label='pymsl')
        plt.legend()

    def calcOldTs(self,method):
        data = []
        time = []
        lat = []
        lon = []
        darrays = []
        for sitei in [self.sites[0]]:
            self.sito=self.specold.defSites(sitei)
            lat.append(sitei.x)
            lon.append(sitei.y)
            for i,S in enumerate(self.specold.readall()):
                time.append(S.time)
                data.append(getattr(S,method)())
            darrays.append(data)
        return xray.DataArray(darrays,[('station',np.arange(0,len(lat))), ('time',time)])


if __name__ == '__main__':
    unittest.main()
    # test = TestSpec()
    # test.setUp()
    # test.test_new()
    # test.test_old()

