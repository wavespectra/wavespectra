"""
Hovmoller diagram of spectra timeseries
=======================================

Integrate spectra and plot as Hovmoller diagram

"""
import matplotlib.pyplot as plt
from wavespectra import read_ww3

fig = plt.figure(figsize=(8, 4))

dset = read_ww3("../_static/ww3file.nc")
ds = (
    dset.isel(site=0)
    .spec.split(fmax=0.18)
    .spec.oned()
    .rename({"freq": "period"})
    .load()
)
ds = ds.assign_coords({"period": 1 / ds.period})
ds.period.attrs.update({"standard_name": "sea_surface_wave_period", "units": "s"})
p = ds.plot.contourf(x="time", y="period", vmax=1.25)
