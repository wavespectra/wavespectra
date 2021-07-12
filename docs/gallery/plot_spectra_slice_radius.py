"""
Control radius extent
=====================

Radius extent can be defined from `rmin` and `rmax` parameters

"""
import matplotlib.pyplot as plt
from wavespectra import read_era5


dset = read_era5("../_static/era5file.nc")
ds = dset.isel(lat=0, lon=0, time=0)
p = ds.spec.plot(
    rmin=0,
    rmax=0.15,
    logradius=False,
    normalised=False,
    levels=25,
    cmap="gray_r",
    radii_ticks=[0.03, 0.06, 0.09, 0.12, 0.15],
    radii_labels=["0.05", "0.1", "0.15Hz"],
    radii_labels_angle=120,
    radii_labels_size=7
)

