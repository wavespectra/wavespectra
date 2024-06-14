"""
Faceting with clean axes
========================

Removing axes could help visualising patterns in multiple plots

"""
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from wavespectra import read_era5


dset = read_era5("../_static/era5file.nc").isel(time=0)
dset1 = dset.where(dset > 0, 1e-5)
dset1 = np.log10(dset1)
p = dset1.spec.plot(
    clean_axis=True,
    col="lon",
    row="lat",
    figsize=(16, 8),
    logradius=False,
    vmin=0.39,
    levels=15,
    extend="both",
    cmap=cmocean.cm.thermal,
    add_colorbar=False,
)
plt.tight_layout()
