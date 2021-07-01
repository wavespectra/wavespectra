import numpy as np
import xarray as xr
from scipy.constants import g, pi

from wavespectra import SpecArray
from wavespectra.core.utils import D2R, R2D, wavenuma, scaled, check_same_coordinates
from wavespectra.core.attributes import attrs, set_spec_attributes


def cartwright(dir, dm, dspr):
    """Cosine-squared directional spreading of Cartwright (1963).

    Args:
        - dir (DataArray): Wave directions (degree).
        - dm: (DataArray, float): Mean wave direction (degree).
        - dspr (DataArray, float) Directional spreading (degree).

    Returns:
        - gth (DataArray): Normalised spreading function.

    Note:
        - If `dm` and `dspr` are DataArrays they must share the same coordinates.

    """
    check_same_coordinates(dm, dspr)

    # Angular difference from mean direction:
    dth = np.abs(dir - dm)
    dth = dth.where(dth <= 180, 360.0 - dth)  # for directional wrapping

    # spread function:
    s = 2.0 / (np.deg2rad(dspr) ** 2) - 1
    gth = np.cos(0.5 * np.deg2rad(dth)) ** (2 * s)
    gth = gth.where(np.abs(dth) <= 90.0, 0.0)  # mask directions +-90 deg from mean

    # normalise
    gsum = 1.0 / (gth.sum(attrs.DIRNAME) * (2 * pi / dir.size))
    gth = gth * gsum

    return gth / R2D


def bunney(dir, dpm, dm, fp, fm):
    """Frequency-dependent assymetrical directional spreading of Bunney et al. (2014).

    Args:
        - dir ():
        - dpm ():
        - fp ():
        - fm ():

    """
    # dtf = (dpm - dm) / (fp - fm)
    pass


# from wavespectra import read_wavespectra
# dset = read_wavespectra("ww3_spec.nc").isel(time=0, site=0, drop=True).efth

# freq = dset.freq.load()
# fp = dset.spec.fp().load()
# fm 
# dpm = dset.spec.dpm().load()

# dfreq = (freq - fp)

# if __name__ == "__main__":

#     import matplotlib.pyplot as plt
#     import cmocean
#     from wavespectra import read_wavespectra

#     dset = read_wavespectra("/source/spec_recon_code/examples/weuro-spec-201201.nc")

#     ds = dset.isel(time=0, site=0, drop=True).load().sortby("dir").drop_dims("fastsite")
#     ds.attrs = {}

#     dm = ds.spec.dm().load()
#     dp = ds.spec.dpm().load()
#     sm = ds.spec.sw().load()
#     sp = ds.spec.swe().load()
