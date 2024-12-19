"""
Logarithmic energy density
==========================

The `as_log10` option to plot the :math:`\log{E_{d}(f,d)}` has been deprecated but similar result 
can be achieved by calculating the :math:`\log{E_{d}(f,d)}` beforehand:

"""

import numpy as np
import matplotlib.pyplot as plt
import cmocean
from wavespectra import read_era5


dset = read_era5("../_static/era5file.nc")
ds = dset.isel(lat=0, lon=0, time=0)
ds = ds.where(ds > 0, 1e-5)  # Avoid infinity values
ds = np.log10(ds)
p = ds.spec.plot(
    as_period=True,
    logradius=False,
    cbar_kwargs={"label": "Normalised $\log{E_{d}(f,d)}$"},
    vmin=0.39,
    levels=15,
    extend="both",
    cmap=cmocean.cm.thermal,
)
