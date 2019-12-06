"""
Control radius extent
=====================

Radius extent can be defined by splitting frequencies

"""
import matplotlib.pyplot as plt
from wavespectra import read_swan

dset = read_swan("../_static/swanfile.spec", as_site=True)
ds = dset.isel(site=0, time=0)
fig = plt.figure(figsize=(6, 4))
p = ds.spec.split(fmin=0, fmax=0.2).spec.plot.contourf(cmap="gray_r")
