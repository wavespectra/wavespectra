#This is to test that new and old parameter functions match

import unittest
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import logging

import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

# New imports
import xarray as xr
from spectra.specarray import SpecArray
from spectra.readspec import read_ww3, read_swan, read_ww3_msl

# Old imports
from pymo.data.spectra_new import WW3NCSpecFile, SwanSpecFile
from pymo.core.basetype import Site

D2R=np.pi/180.

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=10)


class TestSpecSwan(unittest.TestCase):

    def setUp(self):
        print("\n === Testing SWAN  ===")
        testfile = './prelud.spec'
        self.startTime = time.time()
        # New spectra
        startTime = time.time()
        self.specnew = read_swan(testfile)
        # Old spectra
        self.specold=SwanSpecFile(testfile)
        self.sites = self.specold.locations

    def tearDown(self):
        t = time.time() - self.startTime
        print ("%s: %.3f" % (self.id(), t))

    def calcNew(self,spec,method='hs'):
        startTime = time.time()
        self.new = getattr(spec,method)()
        self.ntime = time.time() - startTime
        print ('%s: pyspectra: \t\t %s' % (method, self.ntime))

    def calcOld(self,spec,method='hs'):
        startTime = time.time()
        self.old = self.calcOldTs(spec,method=method)
        self.otime = time.time() - startTime
        print ('%s: pymo: \t\t %s' % (method, self.otime))

    def check_method(self, method):
        print("\nChecking %s:" % method)
        self.calcOld(self.specold,method=method)
        self.calcNew(self.specnew,method=method)
        print ('x times improvement: \t%s' % (self.otime/self.ntime))
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

    def test_dm(self):
        self.check_method('dm')

    def test_dpm(self):
        self.check_method('dpm')

    def test_tm01(self):
        self.check_method('tm01')

    def test_tm02(self):
        self.check_method('tm02')

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
            lat.append(sitei.x)
            lon.append(sitei.y)
            for i,S in enumerate(spec.readall()):
                data.append(getattr(S,method)())

            darrays.append(data)
        time = self.specold.times
        return xr.DataArray(darrays,[('station',np.arange(0,len(lat))), ('time',time)]).transpose()

class TestSpecWW3Native(TestSpecSwan):

    def setUp(self):
        print("\n === Testing WW3 native ===")
        testfile = './snative20141201T00Z_spec.nc'
        self.startTime = time.time()
        # New spectra
        startTime = time.time()
        self.specnew = read_ww3(testfile)
        # Old spectra
        self.specold=WW3NCSpecFile(testfile)
        self.sites = [Site(x=x,y=y) for x,y in zip(self.specold.lon,self.specold.lat)]

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

class TestSpecWW3MSL(TestSpecWW3Native):

    def setUp(self):
        print("\n === Testing WW3 MSL  ===")
        testfile = './med20170319_12z.nc'
        self.startTime = time.time()
        # New spectra
        startTime = time.time()
        self.specnew = read_ww3_msl(testfile)
        # Old spectra
        self.specold=WW3NCSpecFile(testfile)
        self.sites = [Site(x=x,y=y) for x,y in zip(self.specold.lon,self.specold.lat)]

class TestWW3Dims(unittest.TestCase):

    def setUp(self):
        self.ww3msl = read_ww3_msl('./s20170328_12z.nc')
        self.ww3native = read_ww3('./snative20141201T00Z_spec.nc')

    def test_lons(self):
        self.assertEqual(len(self.ww3native.lon.shape), len(self.ww3msl.lon.shape))

    def test_lats(self):
        self.assertEqual(len(self.ww3native.lon.shape), len(self.ww3msl.lon.shape))

if __name__ == '__main__':
    unittest.main()
