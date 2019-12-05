"""
Plot Hs
=======

Plots Hs calculated from spectra dataset

"""
import os
import matplotlib.pyplot as plt

from wavespectra import read_ww3

WAVESPECTRA_DIR = "/source/wavespectra"
dset = read_ww3(os.path.join(WAVESPECTRA_DIR, "tests/sample_files/ww3file.nc"))

fig, ax = plt.subplots(1, 1, figsize=(8,6))

hs = dset.spec.hs()
hs.isel(site=0).plot(ax=ax)
hs.isel(site=1).plot(ax=ax)

plt.tight_layout()
plt.show()