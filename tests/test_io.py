import logging
import unittest
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal

import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from spectra.specarray import SpecDataset
from spectra.io import read_ww3, read_swan, read_ww3_msl, read_netcdf

D2R=np.pi/180.

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=10)

WW3_TEST_FILES=['snative20141201T00Z_spec.nc']
WW3_MSL_TEST_FILES=['s20170328_12z.nc']
SWAN_TEST_FILES=['manus.spec','antf0.20170207_06z.bnd.swn'] #Not yet quite right for hot files,'antf0.20170208_06z.hot-001',]
TMP_DIR='/tmp'

def check_equal(one,other):
    assert_array_almost_equal(one.dset['efth'],other.dset['efth'],decimal=4)
    assert_array_almost_equal(one.dset['freq'],other.dset['freq'],decimal=4)
    assert_array_almost_equal(one.dset['dir'],other.dset['dir'],decimal=4)

class TestSwanIO(unittest.TestCase):
    def setUp(self):
        print("\n === Testing SWAN input and output  ===")
    
    def test_swan_files(self):
        for swanfile in SWAN_TEST_FILES:
            self._test_swan(swanfile)
    
    def _test_swan(self,testfile):
        print ("Reading %s") %testfile
        self.s=read_swan(testfile)
        fileout=os.path.join(TMP_DIR,'test.spec')
        print ("Writing %s") %fileout
        self.s.to_swan(fileout)
        print ("Verifying %s") %fileout
        tmpspec=read_swan(fileout)
        check_equal(self.s,tmpspec)
        
class TestNCIO(unittest.TestCase):
    def setUp(self):
        print("\n === Testing NetCDF input and output  ===")
    
    def test_swan_files(self):
        for swanfile in SWAN_TEST_FILES:
            print ("Reading %s") %swanfile
            self.s=read_swan(swanfile)
            self._test_netcdf(swanfile)
    
    def _test_netcdf(self,testfile):
        fileout=os.path.join(TMP_DIR,os.path.basename(testfile)+'.nc')
        print ("Writing %s") %fileout
        self.s.to_netcdf(fileout,'w')
        self.s.close()
        print ("Verifying %s") %fileout
        tmpspec=read_netcdf(fileout)
        check_equal(self.s,tmpspec)
        
class TestJSONIO(unittest.TestCase):
    def setUp(self):
        print("\n === Testing JSON output  ===")
    
    def test_swan(self):
        for testfile in SWAN_TEST_FILES:
            print ("Reading %s") %testfile
            self.s=read_swan(testfile)
            fileout=os.path.join(TMP_DIR,os.path.basename(testfile)+'.json')
            print ("Writing %s") %fileout
            self.s.to_json(fileout)
            
    def test_ww3(self):
        for testfile in WW3_TEST_FILES:
            print ("Reading %s") %testfile
            self.s=read_ww3(testfile)
            fileout=os.path.join(TMP_DIR,os.path.basename(testfile).replace('.nc','.json'))
            print ("Writing %s") %fileout
            self.s.to_json(fileout)
            
    def test_ww3_msl(self):
        for testfile in WW3_MSL_TEST_FILES:
            print ("Reading %s") %testfile
            self.s=read_ww3_msl(testfile)
            fileout=os.path.join(TMP_DIR,os.path.basename(testfile).replace('.nc','.json'))
            print ("Writing %s") %fileout
            self.s.to_json(fileout)
            
class TestOctopus(unittest.TestCase):
    def setUp(self):
        print("\n === Testing Octopus output  ===")
    
    def test_swan(self):
        for testfile in SWAN_TEST_FILES[0:1]:
            print ("Reading %s") %testfile
            self.s=read_swan(testfile)
            fileout=os.path.join(TMP_DIR,os.path.basename(testfile)+'.oct')
            print ("Writing %s") %fileout
            self.s.to_octopus(fileout)
            
    def test_ww3(self):
        for testfile in WW3_TEST_FILES:
            print ("Reading %s") %testfile
            self.s=read_ww3(testfile)
            fileout=os.path.join(TMP_DIR,os.path.basename(testfile).replace('.nc','.oct'))
            print ("Writing %s") %fileout
            SpecDataset(self.s.isel(site=0)).to_octopus(fileout)
            
    def test_ww3_msl(self):
        for testfile in WW3_MSL_TEST_FILES:
            print ("Reading %s") %testfile
            self.s=read_ww3_msl(testfile)
            fileout=os.path.join(TMP_DIR,os.path.basename(testfile).replace('.nc','.oct'))
            print ("Writing %s") %fileout
            SpecDataset(self.s.isel(site=0)).to_octopus(fileout)
    
if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestOctopus("test_ww3"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
unittest.main()
    