"""
Linear radii
============

Radii are shown on a logarithmic scale by default but can also be shown on a linear scale

"""
import matplotlib.pyplot as plt
from wavespectra import read_era5


dset = read_era5("../_static/era5file.nc")
ds = dset.isel(lat=0, lon=0, time=0)
p = ds.spec.plot(
        as_period=True,
        normalised=False,
        levels=15,
        cmap="bone_r",
        logradius=False,
        radii_ticks=[5, 10, 15, 20, 25],
)
