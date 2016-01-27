"""
Spectra object based on DataArray with methods to calculate spectral statistics

Reference:
- Cartwright and Longuet-Higgins (1956). The Statistical Distribution of the Maxima of a Random Function,
  Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, 237, 212-232.
- Longuet-Higgins (1975). On the joint distribution of the periods and amplitudes of sea waves,
  Journal of Geophysical Research, 80, 2688-2694, doi: 10.1029/JC080i018p02688.
- Zhang (2011).
"""
import copy
from collections import OrderedDict
from datetime import datetime
import numpy as np
import xray as xr # xray.0.6.1
# import xarray as xr # xarray.0.7.0
from timer import Timer


# TODO: We are instantiating the output of slicing operation because they loose dd, df attrs - is there a better way?
# TODO: Currently initializing with an xr.DataArray allows for additional
#       dimensions (e.g. sites), but initializing with np.array does not

# TODO: fix momf() for multi-dimensional array
# TODO: fix momd() for multi-dimensional array
# TODO: fix tp() for multi-dimensional array
# TODO: fix tm01() for multi-dimensional array
# TODO: fix tm02() for multi-dimensional array
#ValueError: coordinate time has dimensions (u'time',), but these are not a subset of the DataArray dimensions ['dim_0', 'dim_1', 'dim_2', 'dim_3']
_ = np.newaxis
gamma = lambda x: np.sqrt(2.*np.pi/x) * ((x/np.exp(1)) * np.sqrt(x*np.sinh(1./x)))**x
# TODO verify which of these are needed...
d2r = np.pi/180.
r2d = 180./np.pi
pi = np.pi
pi2 = 2*pi

class SpecArray(xr.DataArray):
    """
    Multi-dimensional SpecArray object Built on the top of DataArray
    """
    def __init__(self, **kwards):
        """
        ---------------------------
        Required keyword arguments:
        ---------------------------

        (1) xray DataArray:
        -------------------
        spec = SpecArray(data_array=data_array[, dim_map=dim_map])

        - data_array :: DataArray object with spectra
        - dim_map    :: (optional) dictionary to map coordinates ('time', 'freq', 'dir') that must be provided if they
                        are called differently in 'data_array', e.g., {'T': 'time', 'My_Frequencies': 'freq'}

        or

        (2) numpy arrays:
        -----------------
        spec = SpecArray(spec_array=spec_array, freq_array=freq_array[, dir_array=dir_array, time_array=time_array])

        - spec_array :: 1D, 2D or 3D numpy array with spectra. Axes must be ordered as: ([time,]freq[,dir])
        - freq_array :: 1D numpy array with frequencies for spectra
        - dir_array  :: (optional) 1D numpy array with directions for spectra
        - time_array :: (optional) 1D numpy array with datetimes for spectra
        """
        # (1)
        if 'data_array' in kwards and isinstance(kwards['data_array'], xr.DataArray):
            darray = copy.deepcopy(kwards['data_array'])
            if 'dim_map' in kwards and isinstance(kwards['dim_map'], dict):
                darray = darray.rename(kwards['dim_map'])
            assert 'freq' in darray.dims, 'Dimension "freq" not in SpecArray'
        # (2)
        elif 'spec_array' in kwards:
            assert 'freq_array' in kwards, 'freq_array must be provided together with spec_array'
            if 'time_array' not in kwards:
                kwards.update({'time_array': [None]})
                kwards['spec_array'] = np.expand_dims(kwards['spec_array'], 0)
            if 'dir_array' not in kwards:
                kwards.update({'dir_array': [None]})
                kwards['spec_array'] = np.expand_dims(kwards['spec_array'], -1)
            coords = OrderedDict((('time', kwards['time_array']),
                                  ('freq', kwards['freq_array']),
                                  ('dir', kwards['dir_array'])))
            darray = xr.DataArray(data=kwards['spec_array'], coords=coords, name='spec')
        else:
            raise Exception('Either "data_array" or "spec_array" keyword arguments must be provided')

        # Ensure frequencies and directions are sorted
        for dim in ['freq', 'dir']:
            if dim in darray.dims and not self._strictly_increasing(darray[dim].values):
                darray = self.sort(darray, dims=[dim])

        super(SpecArray, self).__init__(data=darray, coords=darray.coords, name='spec')

        self.df = abs(self.freq[1:].values - self.freq[:-1].values) if len(self.freq) > 1 else np.array((1.0,))
        self.dd = abs(self.dir[1].values - self.dir[0].values) if 'dir' in self.dims and len(self.dir) > 1 else 1.0

    def _expand_dim(self, darray):
        """
        Ensures SpecArray has frequency / direction dimensions and increments
        """
        spec_array = darray.values
        spec_coords = OrderedDict((dim, darray[dim].values) for dim in darray.dims)
        for required_dim in ['freq', 'dir']:
            if required_dim not in darray.dims:
                spec_array = np.expand_dims(spec_array, axis=-1)
                spec_coords.update({required_dim: np.array((1,))})
        return SpecArray(data_array=xr.DataArray(spec_array, coords=spec_coords))

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

    def sort(self, darray, dims, inplace=False):
        """
        Sort "darray" along dimensions in "dims" list so that the respective coordinates are sorted
        """
        other = darray if inplace else copy.deepcopy(darray)
        dims = [dims] if not isinstance(dims, list) else dims
        for dim in dims:
            if dim in other.dims:
                if not self._strictly_increasing(darray[dim].values):
                    other = other.isel(**{dim: np.argsort(darray[dim]).values})
            else:
                raise Exception('Dimension %s not in SpecArray' % (dim))
        return SpecArray(data_array=other)

    def split(self, fmin=None, fmax=None, dmin=None, dmax=None):
        """
        Split spectra over freq and/or dir dimensions
        - fmin :: lowest frequency for split spectra, by default min(freq) - interpolates at fmin if fmin not in freq
        - fmax :: highest frequency for split spectra, by default max(freq) - interpolates at fmax if fmax not in freq
        - dmin :: lowest direction to split spectra over, by default min(dir)
        - dmax :: highest direction to split spectra over, by default max(dir)
        """
        # Slice frequencies
        other = self.sel(freq=slice(fmin, fmax))

        # Interpolate at fmin
        if other.freq.min() != self.freq.min() and other.freq.min() > fmin:
            ifreq = np.where(self.freq > fmin)[0][0]
            df = np.diff(self.freq.isel(freq=[ifreq-1, ifreq]))[0]
            Sint = self.isel(freq=[ifreq]) * (fmin - self.freq.isel(freq=[ifreq-1]).values) +\
                self.isel(freq=[ifreq-1]).values * (self.freq.isel(freq=[ifreq]).values - fmin)
            Sint.freq.values = [fmin]
            other = xr.concat([Sint/df, other], dim='freq')

        # Interpolate at fmax
        if other.freq.max() != self.freq.max() and other.freq.max() < fmax:
            ifreq = np.where(self.freq < fmax)[0][-1]
            df = np.diff(self.freq.isel(freq=[ifreq, ifreq+1]))[0]
            Sint = self.isel(freq=[ifreq+1]) * (fmax - self.freq.isel(freq=[ifreq]).values) +\
                self.isel(freq=[ifreq]).values * (self.freq.isel(freq=[ifreq+1]).values - fmax)
            Sint.freq.values = [fmax]
            other = xr.concat([other, Sint/df], dim='freq')

        # Slice directions
        if 'dir' in other.dims and (dmin is not None or dmax is not None):
            other = self.sort(other, dims=['dir']).sel(dir=slice(dmin, dmax))

        return SpecArray(data_array=other)

    def momf(self, mom=0):
        """
        Calculate given frequency moment
        """
        fp = self.freq.values**mom
        mf = (0.5 * self.df[:,_] *
            (fp[1:,_] * self[{'freq': slice(1, None)}] + fp[:-1,_] * self[{'freq': slice(None,-1)}].values))
        return self._expand_dim(mf.sum(dim='freq'))

    def momd(self, mom=0, theta=90., keep_dir=False):
        """
        Directional moment
        """
        cp = np.cos(np.radians(180 + theta - self.dir.values))**mom
        sp = np.sin(np.radians(180 + theta - self.dir.values))**mom
        msin = (self.dd * (self * sp[_,:])).sum(dim='dir')
        mcos = (self.dd * (self * cp[_,:])).sum(dim='dir')
        if keep_dir:
            return self._expand_dim(msin), self._expand_dim(mcos)
        else:
            return msin, mcos

    def oned(self):
        """
        Returns the one-dimensional frequency spectra
        Direction dimension is dropped after integrating
        """
        return SpecArray(data_array=self.dd * self.sum(dim='dir'))

    def _peak(self, arr):
        """
        Returns the index ipeak of largest peak in 1D-array arr
        A peak is found IFF arr(ipeak-1) < arr(ipeak) < arr(ipeak+1)
        """
        ispeak = (np.diff(np.append(arr[0], arr))>0) &\
                 (np.diff(np.append(arr, arr[-1]))<0)
        isort = np.argsort(arr)
        ipeak = np.arange(len(arr))[isort][ispeak[isort]]
        if any(ipeak):
            return ipeak[-1]
        else:
            return None

    def tp(self, smooth=True):
        """
        Peak wave period
        """
        if len(self.freq) < 3:
            return None
        Sf = self.oned()
        ipeak = Sf.argmax(dim='freq')
        # if not ipeak:
        #     return None
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
        # tp = SpecArray(spec_array=1./fp[:,:,_],
        #                freq_array=np.zeros(1),
        #                time_array=self.time)
        ipeak.values[:] = 1
        return ipeak / fp

    def hs(self, fmin=None, fmax=None, times=None, tail=True):
        """
        Spectral significant wave height Hm0
        - fmin  :: lowest frequency to integrate over, by default min(freq)
        - fmax  :: highest frequency to integrate over, by default max(freq)
        - times :: list of datetimes to calculate hs over, by default all times
        - tail  :: fit high-frequency tail
        """
        fmin = fmin or self.freq.min()
        fmax = fmax or self.freq.max()
        times = [times] if not isinstance(times, list) and times is not None else times
        other = self.split(fmin, fmax) if fmin is not None or fmax is not None else self
        Sf = other.oned() if 'dir' in self.dims else copy.deepcopy(self)
        if times:
            Sf = Sf.sel(time=times, method='nearest')
        E = 0.5 * (other.df * (Sf[{'freq': slice(1, None)}] + Sf[{'freq': slice(None, -1)}].values)).sum(dim='freq')
        if tail and other.freq[-1] > 0.333:
            E += 0.25 * Sf[{'freq': -1}].values * other.freq[-1].values
        return 4 * np.sqrt(E)

    def tm01(self, times=None):
        """
        Mean absolute wave period Tm01
        true average period from the 1st spectral moment
        """
        return self.momf(0).sum(dim='dir')/self.momf(1).sum(dim='dir')

    def tm02(self):
        """
        Mean absolute wave period Tm02
        Average period of zero up-crossings (Zhang, 2011)
        """
        return np.sqrt(self.momf(0).sum(dim='dir')/self.momf(2).sum(dim='dir'))

    def dp(self):
        """
        Peak (frequency integrated) wave direction
        """
        ind = self.sum(dim='freq').argmax(dim='dir')
        return self.dir.values[ind]
        # ret = ind.copy()
        # ret[:] = self.dir.values[ind]
        # return ret

    def dpm(self):
        """
        Mean wave direction at peak frequency. Similar to previous code
        but uses the true peak and not simply the maximum value
        """
        #imax=self._peak()
        imax = self._truepeak(sum(self.S, 1))
        if imax:
            moms,momc = self.momd(1)
            dpm = math.atan2(moms[imax][0], momc[imax][0])
            return (270 - r2d*dpm) % 360.
        else:
            return -999

    def dm(self):
        """
        Mean wave direction from the 1st spectral moment
        """
        moms, momc = self.momd(1)
        dpm = np.arctan2(moms.sum(dim='freq'), momc.sum(dim='freq'))
        return (270 - r2d*dpm) % 360.

    def dspr(self):
        """
        Directional wave spreading
        The one-sided directional width of the spectrum
        """
        moms, momc = self.momd(1, keep_dir=True)
        dspr = (2 * r2d**2 * (1 - ((moms.momf()**2 + momc.momf()**2)**0.5 / self.momf(0).oned())))**0.5
        return dspr.sum(dim='freq').sum(dim='dir')

    def swe(self):
        """
        Spectral width parameter by Cartwright and Longuet-Higgins (1956)
        Represents the range of frequencies where the dominant energy exists
        """
        stmp = self.oned()
        swe = (1. - stmp.momf(2).sum()**2/(stmp.momf(0).sum()*stmp.momf(4).sum()))**0.5
        swe.values[swe.values < 0.001] = 1.
        return swe
        # if stmp.hs()<0.001:return 1.
        # return (1. - stmp.momf(2).sum()**2/(stmp.momf(0).sum()*stmp.momf(4).sum()))**0.5;

    def sw(self):
        """
        Spectral width parameter by Longuet-Higgins (1975)
        Represents energy distribution over entire frequency range;
        """
        stmp = self.oned()
        if stmp.hs() < 0.001:
            return 1.
        return (stmp.momf(0).sum() * stmp.momf(2).sum() / stmp.momf(1).sum()**2 - 1.)**0.5



if __name__ == '__main__':
    from pymo.data.spectra import SwanSpecFile
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

    #=================================
    # Real spectra, input as DataArray
    #=================================
    spectra = SwanSpecFile('/Users/rafaguedes/work/prelud0.spec')
    spec_list = [s for s in spectra.readall()]

    spec_array = np.concatenate([np.expand_dims(s.S, 0) for s in spec_list])
    coords=OrderedDict((('dumb_time_name', spectra.times), ('freq', spec_list[0].freqs), ('dir', spec_list[0].dirs)))

    darray = xr.DataArray(data=spec_array, coords=coords)
    spec = SpecArray(data_array=darray, dim_map={'dumb_time_name': 'time'})
    #
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
