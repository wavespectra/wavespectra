"""
Miscellaneous
"""
import datetime
import numpy as np
import pandas as pd

to_datetime = lambda t: datetime.datetime.fromtimestamp(t.astype('int')/1e9)

def spddir_to_uv(spd, direc, coming_from=False):
    """
    converts (spd, dir) to (u, v)
    Input:
        spd :: array with magnitudes to convert
        dired :: array with directions to convert (degree)
        coming_from :: set True if input directions are coming-from convention, False if in going-to
    Output:
        u :: array of same shape as spd and direc with eastward wind component
        v :: array of same shape as spd and direc with northward wind component
    """
    ang_rot = 180 if coming_from else 0
    direc = np.deg2rad(direc % 360)
    ang_rot = np.deg2rad(ang_rot)
    u = spd*np.sin(direc)
    v = spd*np.cos(direc)
    u = u*np.cos(ang_rot) - v*np.sin(ang_rot)
    v = u*np.sin(ang_rot) + v*np.cos(ang_rot)
    return u, v

def uv_to_spddir(u, v, coming_from=False):
    """
    Converts (u, v) to (spd, dir)
    Input:
        u, v :: arrays with eastward and northward wind components
        coming_from :: True for output directions in coming-from convention, False for going-to
    Output:
        mag :: array of same shape as u and v with magnitudes
        direc :: array of same shape as u and v with directions (degree)
    """
    ang_rot = 180 if coming_from else 0
    vetor = u + v*1j
    mag = np.abs(vetor)
    direc = np.angle(vetor)
    direc = direc * 180/np.pi
    direc = direc + ang_rot
    direc = np.mod(90-direc, 360)
    return mag, direc