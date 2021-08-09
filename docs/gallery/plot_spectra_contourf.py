"""
Spectrum as contourf
====================

Contourf type plot of wave spectrum

"""
import matplotlib.pyplot as plt
import cmocean
from wavespectra import read_era5


dset = read_era5("../_static/era5file.nc")
ds = dset.isel(lat=0, lon=0, time=0)
p = ds.spec.plot(kind="contourf", cmap=cmocean.cm.thermal)
