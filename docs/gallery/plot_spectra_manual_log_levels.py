"""
Logarithmic contours
====================

Logarithmic contour levels are only default for normalised spectra but they can be still manually specified

"""
import numpy as np
import matplotlib.pyplot as plt
from wavespectra import read_era5


dset = read_era5("../_static/era5file.nc")
ds = dset.isel(lat=0, lon=0, time=0)
p = ds.spec.plot(
    normalised=False,
    as_period=True,
    cmap="Spectral_r",
    levels=np.logspace(np.log10(0.005), np.log10(0.4), 15),
    cbar_ticks=[0.01, 0.1, 1],
)
