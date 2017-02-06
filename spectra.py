"""
Spectra object based on DataArray with methods to calculate spectral statistics

Reference:
- Cartwright and Longuet-Higgins (1956). The Statistical Distribution of the Maxima of a Random Function,
        Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, 237, 212-232.
- Longuet-Higgins (1975). On the joint distribution of the periods and amplitudes of sea waves,
        Journal of Geophysical Research, 80, 2688-2694, doi: 10.1029/JC080i018p02688.
- Zhang (2011).
"""
from collections import OrderedDict
from datetime import datetime
import numpy as np
import xarray as xr

# TODO: fix momf() for multi-dimensional array
# TODO: fix momd() for multi-dimensional array
# TODO: fix tp() for multi-dimensional array
# TODO: fix tm01() for multi-dimensional array
# TODO: fix tm02() for multi-dimensional array
# TODO: dimension renaming and sorting in __init__ are not producing intended effect. They correctly modify xarray_obj
#       as defined in xarray.spec._obj but the actual xarray is not modified - and they loose their direct association
# TODO: Implement true_peak method for both tp() and dpm()

# Define some globals
GAMMA = lambda x: np.sqrt(2.*np.pi/x) * ((x/np.exp(1)) * np.sqrt(x*np.sinh(1./x)))**x
D2R = np.pi/180.
R2D = 180./np.pi
_ = np.newaxis

@xr.register_dataarray_accessor('spec')
class NewSpecArray(object):
    def __init__(self, xarray_obj, dim_map=None):
        
        # # Rename spectra coordinates if not 'freq' and/or 'dir'
        # if 'dim_map' is not None:
        #     xarray_obj = xarray_obj.rename(dim_map)
        
        # # Ensure frequencies and directions are sorted
        # for dim in ['freq', 'dir']:
        #     if dim in xarray_obj.dims and not self._strictly_increasing(xarray_obj[dim].values):
        #         xarray_obj = self.sort(xarray_obj, dims=[dim])

        self._obj = xarray_obj

        # Create attributes for accessor 
        self.freq = xarray_obj.freq
        self.dir = xarray_obj.dir if 'dir' in xarray_obj.dims else None
        self.df = abs(self.freq[1:].values - self.freq[:-1].values) if len(self.freq) > 1 else np.array((1.0,))
        self.dd = abs(self.dir[1].values - self.dir[0].values) if self.dir is not None and len(self.dir) > 1 else 1.0

    def plot(self):
        """
        Plot spectra
        """
        return 'plotting!'

    def _strictly_increasing(self, arr):
        """
        Returns True if array arr is sorted in increasing order
        """
        return all(x<y for x, y in zip(arr, arr[1:]))

    def _collapse_array(self, arr, indices, axis):
        """
        Collapse n-dim array [arr] along [axis] using [indices]
        """
        magic_index = [np.arange(i) for i in indices.shape]
        magic_index = np.ix_(*magic_index)
        magic_index = magic_index[:axis] + (indices,) + magic_index[axis:]
        return arr[magic_index]

    def _twod(self, darray, dim='dir'):
        """
        Add dir and|or freq dimension if 1D spectra to ensure moment calculations won't break
        """
        if dim not in darray.dims:
            spec_array = np.expand_dims(darray.values, axis=-1)
            spec_coords = OrderedDict((dim, darray[dim].values) for dim in darray.dims)
            spec_coords.update({dim: np.array((1,))})
            return xr.DataArray(spec_array, coords=spec_coords, attrs=darray.attrs)
        else:
            return darray

    def _interp_freq(self, fint):
        """
        Linearly interpolate spectra at frequency fint
        Assumes self.freq.min() < fint < self.freq.max()
        Returns:
            DataArray with one value in frequency dimension (relative to fint) otherwise same dimensions as self._obj
        """
        assert self.freq.min() < fint < self.freq.max(), "Spectra must have frequencies smaller and greater than fint"
        ifreq = self.freq.searchsorted(fint)
        df = np.diff(self.freq.isel(freq=[ifreq-1, ifreq]))[0]
        Sint = self._obj.isel(freq=[ifreq]) * (fint - self.freq.isel(freq=[ifreq-1]).values) +\
            self._obj.isel(freq=[ifreq-1]).values * (self.freq.isel(freq=[ifreq]).values - fint)
        Sint.freq.values = [fint]
        return Sint/df

    def _peak(self, arr):
        """
        Returns the index ipeak of largest peak in 1D-array arr
        A peak is found IFF arr(ipeak-1) < arr(ipeak) < arr(ipeak+1)
        """
        # Temporary - only max values but ensures there is no peak at last value which will break sig2 indexing in tp
        ipeak = arr.argmax(dim='freq')
        return ipeak.where(ipeak < self.freq.size-1).fillna(0).astype(int)

        # ispeak = (np.diff(np.append(arr[0], arr))>0) &\
        #          (np.diff(np.append(arr, arr[-1]))<0)
        # isort = np.argsort(arr)
        # ipeak = np.arange(len(arr))[isort][ispeak[isort]]
        # if any(ipeak):
        #     return ipeak[-1]
        # else:
        #     return None

    def sort(self, darray, dims, inplace=False):
        """
        Sort darray along dimensions in dims list so that the respective coordinates are sorted
        """
        other = darray.copy(deep=not inplace)
        dims = [dims] if not isinstance(dims, list) else dims
        for dim in dims:
            if dim in other.dims:
                if not self._strictly_increasing(darray[dim].values):
                    other = other.isel(**{dim: np.argsort(darray[dim]).values})
            else:
                raise Exception('Dimension %s not in DataArray' % (dim))
        return other

    def oned(self):
        """
        Returns the one-dimensional frequency spectra
        Direction dimension is dropped after integrating
        """
        if self.dir is not None:
            return self.dd * self._obj.sum(dim='dir')
        else:
            return self._obj.copy(deep=True)

    def split(self, fmin=None, fmax=None, dmin=None, dmax=None):
        """
        Split spectra over freq and/or dir dimensions
        - fmin :: lowest frequency to split spectra, by default the lowest - interpolates at fmin if fmin not in freq
        - fmax :: highest frequency to split spectra, by default the highest - interpolates at fmax if fmax not in freq
        - dmin :: lowest direction to split spectra over, by default min(dir)
        - dmax :: highest direction to split spectra over, by default max(dir)
        """
        assert fmax > fmin if fmax else True, 'fmax needs to be greater than fmin'
        assert dmax > dmin if dmax else True, 'fmax needs to be greater than fmin'

        # Slice frequencies
        other = self._obj.sel(freq=slice(fmin, fmax))
        
        # Slice directions
        if 'dir' in other.dims and (dmin is not None or dmax is not None):
            other = self.sort(other, dims=['dir']).sel(dir=slice(dmin, dmax))

        # Interpolate at fmin
        if other.freq.min() > fmin:
            other = xr.concat([self._interp_freq(fmin), other], dim='freq')

        # Interpolate at fmax
        if other.freq.max() < fmax:
            other = xr.concat([other, self._interp_freq(fmax)], dim='freq') 

        return other

    def hs(self, tail=True):
        """
        Spectral significant wave height Hm0
        - tail  :: if True fit high-frequency tail
        """
        Sf = self.oned()
        E = 0.5 * (self.df * (Sf[{'freq': slice(1, None)}] + Sf[{'freq': slice(None, -1)}].values)).sum(dim='freq')
        if tail and Sf.freq[-1] > 0.333:
            E += 0.25 * Sf[{'freq': -1}].values * Sf.freq[-1].values
        return 4 * np.sqrt(E)

    def tp(self, smooth=True, mask=np.nan):
        """
        Peak wave period
        smooth :: if True returns the smooth wave period, if False simply returnsthe discrete
                  period corresponding to the maxima in the direction-integrated spectra
        mask :: value for missing data in output (for when there is no peak in the frequency spectra)
        """
        if len(self.freq) < 3:
            return None
        Sf = self.oned()
        ipeak = self._peak(Sf)
        if smooth:
            freq_axis = Sf.get_axis_num(dim='freq')
            if len(ipeak.dims) > 1:
                sig1 = np.empty(ipeak.shape)
                sig2 = np.empty(ipeak.shape)
                sig3 = np.empty(ipeak.shape)
                for dim in range(ipeak.shape[1]):
                    sig1[:,dim] = self.freq[ipeak[:,dim]-1].values
                    sig2[:,dim] = self.freq[ipeak[:,dim]+1].values
                    sig3[:,dim] = self.freq[ipeak[:,dim]].values
            else:
                sig1 = self.freq[ipeak-1].values
                sig2 = self.freq[ipeak+1].values
                sig3 = self.freq[ipeak].values
            e1 = self._collapse_array(Sf.values, ipeak.values-1, axis=freq_axis)
            e2 = self._collapse_array(Sf.values, ipeak.values+1, axis=freq_axis)
            e3 = self._collapse_array(Sf.values, ipeak.values, axis=freq_axis)
            p = sig1 + sig2
            q = (e1-e2) / (sig1-sig2)
            r = sig1 + sig3
            t = (e1-e3) / (sig1-sig3)
            a = (t-q) / (r-p)
            fp = (-q+p*a) / (2.*a)
            fp[a>=0] = sig3[a>=0]
        else:
            fp = self.freq.values[ipeak]

        # Cheap way to turn Tp into appropriate DataArray object
        ones = ipeak.copy(deep=True) * 0 + 1
        tp = ones / fp

        # Returns masked dataarray
        return tp.where(ipeak>0).fillna(mask)

    def momf(self, mom=0):
        """
        Calculate given frequency moment
        """
        fp = self.freq.values**mom
        mf = (0.5 * self.df[:,_] *
            (fp[1:,_] * self._obj[{'freq': slice(1, None)}] + fp[:-1,_] * self._obj[{'freq': slice(None,-1)}].values))
        return self._twod(mf.sum(dim='freq'))

    def momd(self, mom=0, theta=90.):
        """
        Directional moment
        """
        cp = np.cos(np.radians(180 + theta - self.dir.values))**mom
        sp = np.sin(np.radians(180 + theta - self.dir.values))**mom
        msin = (self.dd * (self._obj * sp[_,:])).sum(dim='dir')
        mcos = (self.dd * (self._obj * cp[_,:])).sum(dim='dir')
        return msin, mcos

    def tm01(self):
        """
        Mean absolute wave period Tm01
        true average period from the 1st spectral moment
        """
        return self.momf(0).sum(dim='dir') / self.momf(1).sum(dim='dir')

    def tm02(self):
        """
        Mean absolute wave period Tm02
        Average period of zero up-crossings (Zhang, 2011)
        """
        return np.sqrt(self.momf(0).sum(dim='dir')/self.momf(2).sum(dim='dir'))

    def dm(self):
        """
        Mean wave direction from the 1st spectral moment
        """
        moms, momc = self.momd(1)
        dpm = np.arctan2(moms.sum(dim='freq'), momc.sum(dim='freq'))
        return (270 - R2D*dpm) % 360.

    def dp(self):
        """
        Peak wave direction defined as the direction where the energy density
        of the frequency-integrated spectrum is maximum
        """
        ind = self._obj.sum(dim='freq').argmax(dim='dir')
        return self.dir.values[ind] * (ind*0+1) # Cheap way to turn dp into appropriate DataArray

    def dpm(self, mask=np.nan):
        """
        From WW3 Manual: peak wave direction, defined like the mean direction, using the
        frequency/wavenumber bin containing of the spectrum F(k) that contains the peak frequency only
        """
        ipeak = self._peak(self.oned())
        moms, momc = self.momd(1)
        moms_peak = self._collapse_array(moms.values, ipeak.values, axis=moms.get_axis_num(dim='freq'))
        momc_peak = self._collapse_array(momc.values, ipeak.values, axis=momc.get_axis_num(dim='freq'))
        dpm = np.arctan2(moms_peak, momc_peak) * (ipeak*0+1) # Cheap way to turn dp into appropriate DataArray
        return ((270 - R2D*dpm) % 360.).where(ipeak>0).fillna(mask)

    def dspr(self):
        """
        Directional wave spreading
        The one-sided directional width of the spectrum
        """
        moms, momc = self.momd(1)
        # Manipulate dimensions so calculations work
        moms = self._twod(moms, dim='dir')
        momc = self._twod(momc, dim='dir')
        mom0 = self._twod(self.momf(0), dim='freq').spec.oned()

        dspr = (2 * R2D**2 * (1 - ((moms.spec.momf()**2 + momc.spec.momf()**2)**0.5 / mom0)))**0.5
        return dspr.sum(dim='freq').sum(dim='dir')

    def swe(self):
        """
        Spectral width parameter by Cartwright and Longuet-Higgins (1956)
        Represents the range of frequencies where the dominant energy exists
        """
        swe = (1. - self.momf(2).sum(dim='dir')**2 / (self.momf(0).sum(dim='dir')*self.momf(4).sum(dim='dir')))**0.5
        swe.values[swe.values < 0.001] = 1.
        return swe

    def sw(self, mask=np.nan):
        """
        Spectral width parameter by Longuet-Higgins (1975)
        Represents energy distribution over entire frequency range;
        """
        sw = (self.momf(0).sum(dim='dir') * self.momf(2).sum(dim='dir') / self.momf(1).sum(dim='dir')**2 - 1.0)**0.5
        return sw.where(self.hs() >= 0.001).fillna(mask)


# lons = np.linspace(1, 5, 5)
# lats = np.linspace(1, 10, 10)
freq = [0.04 * 1.1**n for n in range(10)]
dirs = range(0, 360, 90)
data = np.random.randint(1, 10, (len(freq), len(dirs)))
da = xr.DataArray(data=data, dims=('freq','dir'), coords={'freq': freq, 'dir': dirs})

# class SpecArray(xr.DataArray):
#     """
#     Multi-dimensional SpecArray object Built on the top of DataArray
#     """
#     def __init__(self, **kwards):
#         """
#         ---------------------------
#         Required keyword arguments:
#         ---------------------------

#         (1) xray DataArray:
#         -------------------
#         spec = SpecArray(data_array=data_array[, dim_map=dim_map])

#         - data_array :: DataArray object with spectra
#         - dim_map    :: (optional) dictionary to map coordinates ('time', 'freq', 'dir') that must be provided if they
#                         are called differently in 'data_array', e.g., {'T': 'time', 'My_Frequencies': 'freq'}

#         or

#         (2) numpy arrays:
#         -----------------
#         spec = SpecArray(spec_array=spec_array, freq_array=freq_array[, dir_array=dir_array, time_array=time_array])

#         - spec_array :: 1D, 2D or 3D numpy array with spectra. Axes must be ordered as: ([time,]freq[,dir])
#         - freq_array :: 1D numpy array with frequencies for spectra
#         - dir_array  :: (optional) 1D numpy array with directions for spectra
#         - time_array :: (optional) 1D numpy array with datetimes for spectra
#         """
#         # (1)
#         if 'data_array' in kwards and isinstance(kwards['data_array'], xr.DataArray):
#             darray = copy.deepcopy(kwards['data_array'])
#             if 'dim_map' in kwards and isinstance(kwards['dim_map'], dict):
#                 darray = darray.rename(kwards['dim_map'])
#             assert 'freq' in darray.dims, 'Dimension "freq" not in SpecArray'
#         # (2)
#         elif 'spec_array' in kwards:
#             assert 'freq_array' in kwards, 'freq_array must be provided together with spec_array'
#             if 'time_array' not in kwards:
#                 kwards.update({'time_array': [None]})
#                 kwards['spec_array'] = np.expand_dims(kwards['spec_array'], 0)
#             if 'dir_array' not in kwards:
#                 kwards.update({'dir_array': [None]})
#                 kwards['spec_array'] = np.expand_dims(kwards['spec_array'], -1)
#             coords = OrderedDict((('time', kwards['time_array']),
#                                   ('freq', kwards['freq_array']),
#                                   ('dir', kwards['dir_array'])))
#             darray = xr.DataArray(data=kwards['spec_array'], coords=coords, name='spec')
#         else:
#             raise Exception('Either "data_array" or "spec_array" keyword arguments must be provided')

#         # Ensure frequencies and directions are sorted
#         for dim in ['freq', 'dir']:
#             if dim in darray.dims and not self._strictly_increasing(darray[dim].values):
#                 darray = self.sort(darray, dims=[dim])

#         super(SpecArray, self).__init__(data=darray, coords=darray.coords, name='spec')

#         self.df = abs(self.freq[1:].values - self.freq[:-1].values) if len(self.freq) > 1 else np.array((1.0,))
#         self.dd = abs(self.dir[1].values - self.dir[0].values) if 'dir' in self.dims and len(self.dir) > 1 else 1.0



# if __name__ == '__main__':
#     from pymo.data.spectra import SwanSpecFile
#     #
#     # #=================================
#     # # WW3 spectra, input as DataArray
#     # #=================================
#     # ncfile = 'tests/s20151221_00z.nc'
#     # dset = xr.open_dataset(ncfile)
#     # S = (dset['specden']+127) * dset['factor']
#     # ww3 = SpecArray(data_array=S)
#     #
#     # # hs = ww3.hs()
#     # # tp = ww3.tp()

#     #=================================
#     # Real spectra, input as DataArray
#     #=================================
#     spectra = SwanSpecFile('/Users/rafaguedes/work/prelud0.spec')
#     spec_list = [s for s in spectra.readall()]

#     spec_array = np.concatenate([np.expand_dims(s.S, 0) for s in spec_list])
#     coords=OrderedDict((('dumb_time_name', spectra.times), ('freq', spec_list[0].freqs), ('dir', spec_list[0].dirs)))

#     darray = xr.DataArray(data=spec_array, coords=coords)
#     spec = SpecArray(data_array=darray, dim_map={'dumb_time_name': 'time'})
#     #
#     # hs_new = spec.hs(fmin=0.05, fmax=0.2)
#     # hs_old = [s.split([0.05,0.2]).hs() for s in spec_list]
#     # for old, new, t in zip(hs_old, hs_new, hs_new.time.to_index()):
#     #     print ('Hs old for %s: %0.4f m' % (t, old))
#     #     print ('Hs new for %s: %0.4f m\n' % (t, new))
#     #
#     # print ('Hs for 2015-07-20 18:00:00 (new): %0.3f m' %\
#     #     (spec.hs(fmin=0.05, fmax=0.2, times=datetime(2015,7,20,18), tail=True)))
#     #
#     # #====================================
#     # # Fake spectra, input as numpy arrays
#     # #====================================
#     # freq_array = np.arange(0, 1.01, 0.1)
#     # dir_array = np.arange(0, 360, 30)
#     # time_array = [datetime(2015, 1, d) for d in [1,2,3]]
#     #
#     # # With time and directions
#     # spec_array = np.random.randint(1, 10, (len(time_array), len(freq_array), len(dir_array)))
#     # spec1 = SpecArray(spec_array=spec_array, freq_array=freq_array, dir_array=dir_array, time_array=time_array)
#     #
#     # # Without time
#     # spec_array = np.random.random((len(freq_array), len(dir_array)))
#     # spec2 = SpecArray(spec_array=spec_array, freq_array=freq_array, dir_array=dir_array)
#     #
#     # # Without directions
#     # spec_array = np.random.random((len(time_array), len(freq_array)))
#     # spec3 = SpecArray(spec_array=spec_array, freq_array=freq_array, time_array=time_array)
#     #
#     # # Without time and directions
#     # spec_array = np.random.random(len(freq_array))
#     # spec4 = SpecArray(spec_array=spec_array, freq_array=freq_array)
