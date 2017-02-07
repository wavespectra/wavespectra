"""
Testing new spectra class
"""
from collections import OrderedDict
import numpy as np
import xarray as xr
from pyspectra import NewSpecArray
from pymo.data.spectra import SwanSpecFile

# freq = [0.04 * 1.1**n for n in range(10)]
# dirs = range(0, 360, 90)
# data = np.random.randint(1, 10, (len(freq), len(dirs)))

# da = xr.DataArray(data=data,
#                   dims=('freq','dir'),
#                   coords={'freq': freq, 'dir': dirs})

#=================================
# Real spectra, input as DataArray
#=================================
spectra = SwanSpecFile('./prelud0.spec')
spec_list = [s for s in spectra.readall()]

spec_array = np.concatenate([np.expand_dims(s.S, 0) for s in spec_list])
coords=OrderedDict((('time', spectra.times), ('freq', spec_list[0].freqs), ('dir', spec_list[0].dirs)))
# coords=OrderedDict((('dumb_time_name', spectra.times), ('freq', spec_list[0].freqs), ('dir', spec_list[0].dirs)))

darray = xr.DataArray(data=spec_array, coords=coords)
# da = NewSpecArray(darray, dim_map={'dumb_time_name': 'time'})

# hs_new = darray.spec.split(fmin=0.05, fmax=0.2).spec.hs()
# hs_old = [s.split([0.05,0.2]).hs() for s in spec_list]
# for old, new, t in zip(hs_old, hs_new, hs_new.time.to_index()):
#     print ('Hs old for %s: %0.4f m' % (t, old))
#     print ('Hs new for %s: %0.4f m\n' % (t, new))

new = darray.spec.split(fmin=0.05, fmax=0.07).spec.tp(mask=-999)
old = [s.split([0.05,0.07]).tp() for s in spec_list]
for o, n, t in zip(old, new, new.time.to_index()):
    print ('Tp old for %s: %0.4f m' % (t, o))
    print ('Tp new for %s: %0.4f m\n' % (t, n))

# new = darray.spec.split(fmin=0.05, fmax=0.2).spec.tm01()
# old = [s.split([0.05,0.2]).tm01() for s in spec_list]
# for o, n, t in zip(old, new, new.time.to_index()):
#     print ('Tm01 old for %s: %0.4f m' % (t, o))
#     print ('Tm01 new for %s: %0.4f m\n' % (t, n))

# new = darray.spec.split(fmin=0.05, fmax=0.2).spec.tm02()
# old = [s.split([0.05,0.2]).tm02() for s in spec_list]
# for o, n, t in zip(old, new, new.time.to_index()):
#     print ('Tm02 old for %s: %0.4f m' % (t, o))
#     print ('Tm02 new for %s: %0.4f m\n' % (t, n))

# new = darray.spec.split(fmin=0.05, fmax=0.2).spec.dm()
# old = [s.split([0.05,0.2]).dm() for s in spec_list]
# for o, n, t in zip(old, new, new.time.to_index()):
#     print ('dm old for %s: %0.4f m' % (t, o))
#     print ('dm new for %s: %0.4f m\n' % (t, n))

# new = darray.spec.split(fmin=0.05, fmax=0.2).spec.dspr()
# old = [s.split([0.05,0.2]).dspr() for s in spec_list]
# for o, n, t in zip(old, new, new.time.to_index()):
#     print ('dspr old for %s: %0.4f m' % (t, o))
#     print ('dspr new for %s: %0.4f m\n' % (t, n))

# new = darray.spec.split(fmin=0.05, fmax=0.2).spec.swe()
# old = [s.split([0.05,0.2]).swe() for s in spec_list]
# for o, n, t in zip(old, new, new.time.to_index()):
#     print ('swe old for %s: %0.4f m' % (t, o))
#     print ('swe new for %s: %0.4f m\n' % (t, n))

# new = darray.spec.split(fmin=0.05, fmax=0.2).spec.sw()
# old = [s.split([0.05,0.2]).sw() for s in spec_list]
# for o, n, t in zip(old, new, new.time.to_index()):
#     print ('sw old for %s: %0.4f m' % (t, o))
#     print ('sw new for %s: %0.4f m\n' % (t, n))

# new = darray.spec.split(fmin=0.05, fmax=0.2).spec.dp()
# old = [s.split([0.05,0.2]).dp() for s in spec_list]
# for o, n, t in zip(old, new, new.time.to_index()):
#     print ('dp old for %s: %0.4f m' % (t, o))
#     print ('dp new for %s: %0.4f m\n' % (t, n))

# new = darray.spec.split(fmin=0.05, fmax=0.2).spec.dpm(mask=-999)
# old = [s.split([0.05,0.2]).dpm() for s in spec_list]
# for o, n, t in zip(old, new, new.time.to_index()):
#     print ('dpm old for %s: %0.4f m' % (t, o))
#     print ('dpm new for %s: %0.4f m\n' % (t, n))
#     break



# from pymo.data.spectra import SwanSpecFile
#
# #=================================
# # WW3 spectra, input as DataArray
# #=================================
# ncfile = 'tests/s20151221_00z.nc'
# dset = xr.open_dataset(ncfile)
# S = (dset['specden']+127) * dset['factor']
# ww3 = SpecArray(data_array=S)
#
# # hs = ww3.hs()
# # tp = ww3.tp()

# #=================================
# # Real spectra, input as DataArray
# #=================================
# spectra = SwanSpecFile('/Users/rafaguedes/work/prelud0.spec')
# spec_list = [s for s in spectra.readall()]

# spec_array = np.concatenate([np.expand_dims(s.S, 0) for s in spec_list])
# coords=OrderedDict((('dumb_time_name', spectra.times), ('freq', spec_list[0].freqs), ('dir', spec_list[0].dirs)))

# darray = xr.DataArray(data=spec_array, coords=coords)
# spec = SpecArray(data_array=darray, dim_map={'dumb_time_name': 'time'})

# hs_new = spec.hs(fmin=0.05, fmax=0.2)
# hs_old = [s.split([0.05,0.2]).hs() for s in spec_list]
# for old, new, t in zip(hs_old, hs_new, hs_new.time.to_index()):
#     print ('Hs old for %s: %0.4f m' % (t, old))
#     print ('Hs new for %s: %0.4f m\n' % (t, new))
#
# print ('Hs for 2015-07-20 18:00:00 (new): %0.3f m' %\
#     (spec.hs(fmin=0.05, fmax=0.2, times=datetime(2015,7,20,18), tail=True)))
#
# #====================================
# # Fake spectra, input as numpy arrays
# #====================================
# freq_array = np.arange(0, 1.01, 0.1)
# dir_array = np.arange(0, 360, 30)
# time_array = [datetime(2015, 1, d) for d in [1,2,3]]
#
# # With time and directions
# spec_array = np.random.randint(1, 10, (len(time_array), len(freq_array), len(dir_array)))
# spec1 = SpecArray(spec_array=spec_array, freq_array=freq_array, dir_array=dir_array, time_array=time_array)
#
# # Without time
# spec_array = np.random.random((len(freq_array), len(dir_array)))
# spec2 = SpecArray(spec_array=spec_array, freq_array=freq_array, dir_array=dir_array)
#
# # Without directions
# spec_array = np.random.random((len(time_array), len(freq_array)))
# spec3 = SpecArray(spec_array=spec_array, freq_array=freq_array, time_array=time_array)
#
# # Without time and directions
# spec_array = np.random.random(len(freq_array))
# spec4 = SpecArray(spec_array=spec_array, freq_array=freq_array)