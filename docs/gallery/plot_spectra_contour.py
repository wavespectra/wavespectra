"""
Spectrum as contour
===================

Contour type plot of wave spectrum

"""
import matplotlib.pyplot as plt
from wavespectra import read_era5


dset = read_era5("../_static/era5file.nc")
ds = dset.isel(lat=0, lon=0, time=0)
p = ds.spec.plot(kind="contour", colors="#af1607", linewidths=0.5)
