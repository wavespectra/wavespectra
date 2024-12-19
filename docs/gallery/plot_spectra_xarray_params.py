"""
Plotting parameters from xarray
===============================

Wavespectra allows passing some parameters from the functions wrapped from xarray such as `contourf <http://xarray.pydata.org/en/stable/generated/xarray.plot.contourf.html>`_ 
(excluding some that are manipulated in wavespectra such as `ax`, `x` and others):

"""

import matplotlib.pyplot as plt
from wavespectra import read_era5


dset = read_era5("../_static/era5file.nc")
ds = dset.isel(lat=0, lon=0, time=0)
p = ds.spec.plot(
    kind="contourf", cmap="turbo", add_colorbar=False, extend="both", levels=25
)
