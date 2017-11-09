from ..attributes import *
import numpy as np
import xarray as xr
from collections import OrderedDict

d2r = np.pi/180

#Generic spreading function such that \int{G1 d\theta}=1*
def spread(dp_matrix,dspr_matrix,dirs):
    adirs = np.array(dirs).reshape((1,-1))
    pidirs = d2r*(270.-np.array(adirs))
    st1 = np.sin(0.5*d2r*(270.-dp_matrix))
    ct1 = np.cos(0.5*d2r*(270.-dp_matrix))
    a = np.maximum(np.cos(0.5*pidirs)*ct1+np.sin(0.5*pidirs)*st1,0.0)
    G1 = a**(2.*dspr_matrix)
    G1 /= np.expand_dims(G1.sum(axis=-1)*abs(dirs[1]-dirs[0]),axis=-1)
    return G1

#Check all inputs are same shape and add frequency and direction dims
def arrange_inputs(*args):
    argout = []
    shape0 = np.array(args[0]).shape
    for arg in args:
        argm = np.array(arg)
        if (argm.shape==()) and shape0!=():#Broadcast scalar across matrix
            argm=arg*np.ones(shape0)
        elif argm.shape!=shape0:
            raise 'Input shapes must be the same'
        argout.append(argm[...,np.newaxis,np.newaxis])
    return argout

#Package spectral matrix to xarray
def make_dataset(spec,freqs,dirs,coordinates=[]):
    coords = tuple(coordinates)+((FREQNAME, freqs), (DIRNAME, dirs),)
    dimensions = tuple([c[0] for c in coords])
    dset = xr.DataArray(
            data=spec,
            coords=coords,
            dims=dimensions,
            name=SPECNAME,
            ).to_dataset()
    set_spec_attributes(dset)
    return dset

#Check coordinates are consistent with parameter
def check_coordinates(param,coordinates):
    pshape = np.array(param).shape
    if len(pshape) != len(coordinates):
        raise 'Incorrect number of coordinates for parameter'
    for idim,dim in enumerate(pshape):
        if dim != len(coordinates[idim][1]):
            raise 'Dimension of coordinate %s at position %d does not match parameter' %(coordinates[idim][0],dim)