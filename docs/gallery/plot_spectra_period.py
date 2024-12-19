"""
Wave period spectrum
====================

Plot of a wave spectrum in period space

"""

import matplotlib.pyplot as plt
import cmocean
from wavespectra import read_era5


dset = read_era5("../_static/era5file.nc")
ds = dset.isel(lat=0, lon=0, time=0)
fig = plt.figure(figsize=(6, 4))
p = ds.spec.plot(as_period=True, cmap="pink_r")
