import sys
import os
import logging
import unittest
import time
import numpy as np
from numpy.testing import assert_array_almost_equal

plot = False
if plot:
    import matplotlib.pyplot as plt

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from wavespectra.construct import jonswap, ochihubble

def check_equal(one, other):
    assert_array_almost_equal(one['efth'], other['efth'], decimal=4)
    assert_array_almost_equal(one['freq'], other['freq'], decimal=4)
    assert_array_almost_equal(one['dir'], other['dir'], decimal=4)

#Numerical integration of 2D spectrum
def integrate_2d_hs(freqs, dirs, S):
    sum1 = np.trapz(S, freqs, axis=0)
    sum2 = np.trapz(sum1, dirs)
    return 4 * sum2**0.5

class TestJonswap(unittest.TestCase):
    def setUp(self):
        print("\n === Testing JONSWAP construct  ===")

    def hs(self, tp, alpha, gamma=3.3, df=0.02):
        #Calculate 1D JONSWAP
        f = np.arange(df, 1.0, df)
        fp = 1.0 / tp
        sig = np.where(f<=fp, 0.07, 0.09)
        r = np.exp(-(f-fp)**2. / (2 * sig**2 * fp**2))
        S = 0.0617 * np.array(alpha) * f**(-5) * np.exp(-1.25*(f/fp)**(-4)) * gamma**r
        return 4 * (S.sum() * df)**0.5

    def test_jonswap_scalar(self):
        dset = jonswap(tp=10,dp=90,alpha=0.01)
        if plot:
            plt.pcolormesh(dset['freq'][:], dset['dir'][:], dset['efth'][:,:].T)
            plt.show()
        assert_array_almost_equal(integrate_2d_hs(dset['freq'][:],
                                                  dset['dir'][:],
                                                  dset['efth'][:,:]),
                                  self.hs(10, 0.01), decimal=3)

    def test_jonswap_series(self):
        tp = [10,5]
        dp = [90,180]
        dspr = [25,40]
        dset = jonswap(tp, dp, alpha=0.01, dspr=dspr, coordinates=[('time', [0,0])])  
        if plot:
            plt.pcolormesh(dset['freq'][:], dset['dir'][:], dset['efth'][0,:,:].T)
            plt.show()
            plt.pcolormesh(dset['freq'][:], dset['dir'][:], dset['efth'][1,:,:].T)
            plt.show()

        for i,spec in enumerate(dset['efth']):
            assert_array_almost_equal(integrate_2d_hs(dset['freq'][:],
                                                      dset['dir'][:],
                                                      spec[:,:]),
                                      self.hs(tp[i],0.01), decimal=3)

    def test_jonswap_matrix(self):
        tp = 10 * np.random.random((5, 4, 3))
        dp = 360 * np.random.random((5, 4, 3))
        dset = jonswap(tp, dp, alpha=0.01, dspr=25, coordinates=[('time', np.arange(0, 5)),
                                                                 ('lat', np.arange(0, 4)),
                                                                 ('lon', np.arange(0, 3))
                                                                 ])  
        if plot:
            plt.pcolormesh(dset['freq'][:], dset['dir'][:], dset['efth'][0,:,:].T)
            plt.show()
            plt.pcolormesh(dset['freq'][:], dset['dir'][:], dset['efth'][1,:,:].T)
            plt.show()

        i = np.random.randint(5)
        j = np.random.randint(4)
        k = np.random.randint(3)

        assert_array_almost_equal(integrate_2d_hs(dset['freq'][:],
                                                  dset['dir'][:],
                                                  dset['efth'][i,j,k,:,:]),
                                  self.hs(tp[i,j,k], 0.01), decimal=3)

class TestOchiHubble(unittest.TestCase):
    def setUp(self):
        print("\n === Testing OchiHubble construct  ===")

    def hs(self,hs,tp,l,df=0.02):
        #Calculate 1D OH
        gamma = lambda x: np.sqrt(2.*np.pi/x) * ((x/np.exp(1.)) * np.sqrt(x*np.sinh(1./x)))**x
        w = 2 * np.pi * np.arange(df, 1.0, df)
        S = np.zeros((len(w)))
        for i,H in enumerate(hs):
            #Create 1D spectrum
            w0 = 2 * np.pi / tp[i]
            B = np.maximum(l[i], 0.01) + 0.25
            A = 0.5 * np.pi * H**2 * ((B*w0**4)**l[i] / gamma(l[i]))
            a = np.minimum((w0 / w)**4, 100.)
            S += A * np.exp(-B*a) / (np.power(w, 4.*B))
        return 4 * (S.sum() * df)**0.5

    def test_oh_scalar(self):
        """Test single set of parameters."""
        dset = ochihubble(hs=[1.0,1.0], tp=[10,5], L=[1,3], dp=[90,180], dspr=[25,40])
        if plot:
            plt.pcolormesh(dset['freq'][:], dset['dir'][:], dset['efth'][:,:].T)
            plt.show()
        assert_array_almost_equal(integrate_2d_hs(dset['freq'][:],
                                                  dset['dir'][:],
                                                  dset['efth'][:,:]),
                                  self.hs([1.,1.], [10,5], [1,3]), decimal=3)

    def test_oh_series(self):
        """Test 1D arrays of parameters."""
        hs = [1.0 * np.random.random(10), 1.0*np.random.random(10)]
        tp = [10.0 * np.random.random(10) + 1, 10.0*np.random.random(10) + 1]
        L = [1.0 * np.random.random(10) + 1, 3.0*np.random.random(10) + 1]
        dp = [360.0 * np.random.random(10), 360.0*np.random.random(10)]
        dspr = [20.0 * np.random.random(10) + 10., 50.0*np.random.random(10) + 10.]
        dset = ochihubble(hs, tp, L, dp, dspr, coordinates=[('time', np.arange(0,10))])
        if plot:
            plt.pcolormesh(dset['freq'][:], dset['dir'][:], dset['efth'][0,:,:].T)
            plt.show()
            plt.pcolormesh(dset['freq'][:], dset['dir'][:], dset['efth'][1,:,:].T)
            plt.show()

        for i,spec in enumerate(dset['efth']):
            params = [
                [h[i] for h in hs],
                [t[i] for t in tp],
                [l[i] for l in L]
            ]
            print(params)
            assert_array_almost_equal(integrate_2d_hs(dset['freq'][:],
                                                      dset['dir'][:],
                                                      spec[:,:]),
                                      self.hs(*params), decimal=3)

if __name__ == '__main__':
    unittest.main()
    