"""Miscellaneous functions."""
import copy
import datetime
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import griddata

GAMMA = lambda x: np.sqrt(2.*np.pi/x) * ((x/np.exp(1)) * np.sqrt(x*np.sinh(1./x)))**x
D2R = np.pi / 180.
R2D = 180. / np.pi

dnum_to_datetime = lambda d: datetime.datetime.fromordinal(int(d) - 366) + datetime.timedelta(days=d%1)

to_nautical = lambda a: np.mod(270-a, 360)

def to_datetime(np64):
    """Convert Datetime64 date to datatime."""
    if isinstance(np64, np.datetime64):
        dt = pd.to_datetime(str(np64)).to_pydatetime()
    elif isinstance(np64, xr.DataArray):
        dt = pd.to_datetime(str(np64.values)).to_pydatetime()
    else:
        IOError('Cannot convert %s into datetime, expected np.datetime64 or xr.DataArray' %
            type(np64))
    return dt

def spddir_to_uv(spd, direc, coming_from=False):
    """Converts (spd, dir) to (u, v).

    Args:
        spd (array): magnitudes to convert
        direc (array): directions to convert (degree)
        coming_from (bool): True if directions in coming-from convention, False if in going-to

    Returns:
        u (array): eastward wind component
        v (array): northward wind component

    """
    ang_rot = 180 if coming_from else 0
    direcR = np.deg2rad(direc + ang_rot)     
    u = spd * np.sin(direcR)
    v = spd * np.cos(direcR)
    return u, v

def uv_to_spddir(u, v, coming_from=False):
    """Converts (u, v) to (spd, dir).

    Args:
        u (array): eastward wind component
        v (array): northward wind component
        coming_from (bool): True for output directions in coming-from convention, False for going-to

    Returns:
        mag (array): magnitudes
        direc (array): directions (degree)
    """
    ang_rot = 180 if coming_from else 0
    vetor = u + v*1j
    mag = np.abs(vetor)
    direc = np.angle(vetor, deg=True)
    direc = direc + ang_rot
    direc = np.mod(90-direc, 360)
    return mag, direc

def interp_spec(inspec, infreq, indir, outfreq=None, outdir=None, method='linear'):
    """Interpolate onto new spectral basis.

    Args:
        inspec (2D ndarray): input spectrum E(infreq,indir) to be interpolated
        infreq (1D ndarray): frequencies of input spectrum
        indir (1D ndarray): directions of input spectrum
        outfreq (1D ndarray): frequencies of output interpolated spectrum, same as infreq by default
        outdir (1D ndarray): directions of output interpolated spectrum, same as infreq by default
        method: {'linear', 'nearest', 'cubic'}, method of interpolation to use with griddata

    Returns:
        outspec (2D ndarray): interpolated ouput spectrum E(outfreq,outdir)

    Note:
        If either outfreq or outdir is None or False this coordinate is not interpolated
        Choose indir=None if spectrum is 1D

    """
    outfreq = infreq if outfreq is None or outfreq is False else outfreq
    outdir = indir if outdir is None or outdir is False else outdir
    
    if (np.array_equal(infreq, outfreq)) & (np.array_equal(indir, outdir)):
        outspec = copy.deepcopy(inspec)
    elif np.array_equal(indir, outdir):
        if indir is not None:
            outspec = np.zeros((len(outfreq), len(outdir)))            
            for idir in range(len(indir)):
                outspec[:,idir] = np.interp(outfreq, infreq, inspec[:,idir], left=0., right=0.)
        else:
            outspec = np.interp(outfreq, infreq, np.array(inspec).ravel(), left=0., right=0.)
    else:
        dirs = D2R * (270 - np.expand_dims(outdir,0))
        dirs2 = D2R * (270 - np.expand_dims(indir,0))
        cosmat = np.dot(np.expand_dims(outfreq,-1), np.cos(dirs))
        sinmat = np.dot(np.expand_dims(outfreq,-1), np.sin(dirs))
        cosmat2 = np.dot(np.expand_dims(infreq,-1), np.cos(dirs2))
        sinmat2 = np.dot(np.expand_dims(infreq,-1), np.sin(dirs2))
        outspec = griddata((cosmat2.flat, sinmat2.flat), inspec.flat, (cosmat,sinmat), method, 0.)
    return outspec

def flatten_list(l, a):
    """Flatten list of lists"""
    for i in l:
        if isinstance(i, list):
            flatten_list(i, a)
        else:
            a.append(i)
    return a