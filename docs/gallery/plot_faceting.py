"""
Faceting capability
===================

Faceting capability from xarray is supported

"""
import matplotlib.pyplot as plt
from wavespectra import read_swan

dset = read_swan("../_static/swanfile.spec", as_site=True)
p = dset.isel(site=0).spec.plot.contourf(
    col="time", col_wrap=3, levels=15, figsize=(15, 8), vmax=-1, cmap="jet"
)
