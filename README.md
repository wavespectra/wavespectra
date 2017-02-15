# pyspectra
Python library for wave spectra

### Example to define and plot spectra history from SWAN bnd spectra file:
```python
from pyspectra.iospec import read_spec_swan

dset = read_spec_swan('/source/pyspectra/pyspectra/tests/antf0.20170207_06z.bnd.swn')
spec_hist = dset['efth'].isel(site=0).spec.oned().T
spec_hist.plot()
```
