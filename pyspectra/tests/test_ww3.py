import logging
import unittest
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

# New imports
import xarray as xr
from pyspectra.spectra import NewSpecArray
from pyspectra.iospec import read_spec_ww3_native, SPECNAME

# Old imports
from pymo.data.spectra_new import WW3NCSpecFile
from pymo.core.basetype import Site

# testfile = '/home/tdurrant/Downloads/shell_spec_test.nc'
#testfile = '/home/tdurrant/Documents/projects/shell/code/old/shell_spec_test.nc'
testfile = './snative20141201T00Z_spec.nc'

D2R=np.pi/180.

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=10)


class TestSpec(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()
        # New spectra
        startTime = time.time()
        self.specnew = read_spec_ww3_native(testfile)
        # Old spectra
        self.specold=WW3NCSpecFile(testfile)
        self.sites = [Site(x=x,y=y) for x,y in zip(self.specold.lon,self.specold.lat)]

    def tearDown(self):
        t = time.time() - self.startTime
        print ("%s: %.3f" % (self.id(), t))

    def calcNew(self,spec,method='hs'):
        startTime = time.time()
        self.new = getattr(spec,method)()
        self.ntime = time.time() - startTime
        print ('%s: pymsl: %s' % (method, self.ntime))

    def calcOld(self,spec,method='hs'):
        startTime = time.time()
        self.old = self.calcOldTs(spec,method=method)
        self.otime = time.time() - startTime
        print ('%s: pymo: %s' % (method, self.otime))

    def check_method(self,method):
        print("\nChecking %s:" % method)
        self.calcOld(self.specold,method=method)
        self.calcNew(self.specnew[SPECNAME].spec,method=method)
        print ('x times improvement: %s' % (self.otime/self.ntime))
        assert_array_almost_equal(self.old.values.squeeze(),self.new.values.squeeze(), decimal=4,)
        # self.plot()
        # plt.title(method)
        # plt.show()

    def test_hs(self):
        self.check_method('hs')

    def test_tp(self):
        self.check_method('tp')

    def test_dspr(self):
        self.check_method('dspr')

#    def test_sea(self,split=9.):
#        startTime = time.time()
#        nsw = self.specnew.split(fmin=1./split)
#        nsea = self.specnew.split(fmax=1./split)
#        print ('pymsl split: %s' % (time.time() - startTime))
#        startTime = time.time()
#        osw = self.specold.split(fmin=1./split)
#        osea = self.specold.split(fmax=1./split)
#        print ('pymo split: %s' % (time.time() - startTime))
#        for method in ('hs',):
#            self.calcOld(nsea,method=method)
#            self.calcNew(osea,method=method)
#            self.plot()
#            plt.title(method+' Sea')
#            self.calcOld(nsw,method=method)
#            self.calcNew(osw,method=method)
#            self.plot()
#            plt.title(method+' Sea')
#            #np.allclose(self.old.values, self.new.values)
#        plt.show()

    def plot(self):
        plt.figure()
        self.old.plot(label='pymo')
        plt.figure()
        self.new.plot(label='pymsl')
        #plt.legend()

    def calcOldTs(self,spec,method):
        darrays = []
        lat = []
        lon = []
        for sitei in self.sites:
        #for sitei in [self.sites[0]]:
            data = []
            time = []
            self.sito=spec.defSites(sitei)
            lat.append(sitei.x)
            lon.append(sitei.y)
            for i,S in enumerate(spec.readall()):
                time.append(S.time)
                data.append(getattr(S,method)())
            darrays.append(data)
        return xr.DataArray(darrays,[('station',np.arange(0,len(lat))), ('time',time)]).transpose()

if __name__ == '__main__':
    unittest.main()
