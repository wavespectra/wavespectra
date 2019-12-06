"""
Period spectrum
===============

Plot of a wave spectrum in period space

"""
import matplotlib.pyplot as plt
from wavespectra import read_swan

dset = read_swan("../_static/swanfile.spec", as_site=True)
ds = dset.isel(site=0, time=0)
fig = plt.figure(figsize=(6, 4))
p = ds.spec.plot.contourf(as_period=True)
