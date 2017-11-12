import xarray as xr
import numpy as np
from helpers import *

gamma = lambda x:np.sqrt(2.*np.pi/x)*((x/np.exp(1.))*np.sqrt(x*np.sinh(1./x)))**x

#OchiHubble construct function
def ochihubble(hs,tp,L,dp,dspr,freqs=np.arange(0.02,1.0,0.02),dirs=np.arange(0,360,10),coordinates=[]):
    hs0 = hs
    for inp in [hs,tp,L,dp,dspr]:
        if not isinstance(inp,list):
            raise 'OchiHubble input parameters must be lists'
        elif len(inp)!=len(hs0):
            raise 'All inputs must same number of members'    
    check_coordinates(hs[0],coordinates)
    hs_m = []
    tp_m = []
    l_m = []
    dp_m = []
    dspr_m = []
    for i,h in enumerate(hs):
        #arrange all input parameters onto consistent matrices
        hs_tmp,tp_tmp,l_tmp,dp_tmp,dspr_tmp = arrange_inputs(h,tp[i],L[i],dp[i],dspr[i])
        hs_m.append(hs_tmp)
        tp_m.append(tp_tmp)
        l_m.append(l_tmp)
        dp_m.append(dp_tmp)
        dspr_m.append(dspr_tmp)
    hs_m0 = hs_m[0]
    for hs_m1 in hs_m:
        if hs_m1.shape != hs_m0.shape:
            raise 'Shapes of parameter arguments must match'
    #Create spectra matrix
    spec = np.zeros(hs_m0.shape[:-2]+(len(freqs),len(dirs),))
    S = np.zeros((len(freqs),1))
    w = 2*np.pi*freqs.reshape((-1,1))

    #Loop over partitions
    for i,H in enumerate(hs_m):
        #Create 1D spectrum
        w0 = 2*np.pi/tp_m[i]
        B = np.maximum(l_m[i],0.01)+0.25
        A = 0.5*np.pi*H**2*((B*w0**4)**l_m[i]/gamma(l_m[i]))
        a = np.minimum((w0/w)**4,100.)
        S = A*np.exp(-B*a)/(np.power(w,4.*B))

        #Apply spreading
        G1 = spread(dp_m[i],dspr_m[i],dirs)
        spec += S*G1
        
    return make_dataset(spec,freqs,dirs,coordinates)



