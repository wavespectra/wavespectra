# pyspectra
Python library for wave spectra

### Example to define and plot spectra history from SWAN bnd spectra file:
```python
from spectra import read_swan

dset = read_swan('/source/pyspectra/tests/antf0.20170207_06z.bnd.swn')
spec_hist = dset.efth.isel(site=0).spec.oned().T
spec_hist.plot()
```
