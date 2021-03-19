"""
Spectrum as contour
===================

Contour type plot of wave spectrum

"""
import matplotlib.pyplot as plt
from wavespectra import read_swan

dset = read_swan("../_static/swanfile.spec", as_site=True)
ds = dset.isel(site=0, time=0)
fig = plt.figure(figsize=(6, 4))
p = ds.spec.plot.contour()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(ds.dir, ds.spec.sum('freq').efth)
plt.title('Per direction')
plt.show()
