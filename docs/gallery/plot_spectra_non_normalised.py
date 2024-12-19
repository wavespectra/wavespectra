"""
Energy density values
=====================

Show actual energy density rather than normalised values

"""

import matplotlib.pyplot as plt
from wavespectra import read_era5


dset = read_era5("../_static/era5file.nc")
ds = dset.isel(lat=0, lon=0, time=0)
p = ds.spec.plot(
    normalised=False,
    as_period=True,
    cmap="Spectral_r",
    levels=10,
)
