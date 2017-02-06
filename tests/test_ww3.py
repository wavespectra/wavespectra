import sys
sys.path.append('../')

import logging
import unittest
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal

reload(sys)
sys.setdefaultencoding('utf-8')

# New imports
import xray
from spectra import SpecArray

# Old imports
from pymo.data.spectra_new import WW3NCSpecFile
from pymo.core.basetype import Site

# testfile = '/home/tdurrant/Downloads/shell_spec_test.nc'
testfile = '/home/tdurrant/Documents/projects/shell/code/old/shell_spec_test.nc'

D2R=np.pi/180.

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=10)


class TestSpec(unittest.TestCase):
# class TestSpec(object):

    def setUp(self):
        self.startTime = time.time()
        # New spectra
        startTime = time.time()
        dset = xray.open_dataset(testfile)
        coords={'direction': 'dir', 'frequency':'freq'}
        dset.efth *= D2R
        dset.direction += 180
        dset.direction %= 360
        self.specnew = SpecArray(data_array=dset.efth[:,0],dim_map=coords)
        #self.specnew = SpecArray(data_array=dset.efth,dim_map=coords)
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

    def test_vars(self):
        #for method in ('hs','tm01','tm02','dm'):
        for method in ('tp',):
            self.calcOld(self.specold,method=method)
            self.calcNew(self.specnew,method=method)
            print ('x times improvement: %s' % (self.otime/self.ntime))
            #assert_array_almost_equal(self.old.values.squeeze(),self.new.values.squeeze())
            self.plot()
            plt.title(method)
        plt.show()

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
        self.new.plot(label='pymsl')
        plt.legend()

    def calcOldTs(self,spec,method):
        darrays = []
        lat = []
        lon = []
        #for sitei in self.sites:
        for sitei in [self.sites[0]]:
            data = []
            time = []
            self.sito=spec.defSites(sitei)
            lat.append(sitei.x)
            lon.append(sitei.y)
            for i,S in enumerate(self.specold.readall()):
                time.append(S.time)
                data.append(getattr(S,method)())
            darrays.append(data)
        #npdat = np.array(darrays)
        #return xray.DataArray(npdat.transpose(),[('time',time),('station',np.arange(0,len(lat)))])
        return xray.DataArray(darrays,[('station',np.arange(0,len(lat))), ('time',time)])


if __name__ == '__main__':
    unittest.main()
    #test = TestSpec()
    #test.setUp()
    #test.test_new()
    #test.test_old()

