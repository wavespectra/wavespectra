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
import types
import copy

# TODO: dimension renaming and sorting in __init__ are not producing intended effect. They correctly modify xarray_obj
#       as defined in xarray.spec._obj but the actual xarray is not modified - and they loose their direct association
# TODO: Implement true_peak method for both tp() and dpm()

# Define some globals
GAMMA = lambda x: np.sqrt(2.*np.pi/x) * ((x/np.exp(1)) * np.sqrt(x*np.sinh(1./x)))**x
D2R = np.pi/180.
R2D = 180./np.pi
_ = np.newaxis

@xr.register_dataarray_accessor('spec')
class SpecArray(object):
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
            return xr.DataArray(spec_array, coords=spec_coords, dims=spec_coords.keys(), attrs=darray.attrs)
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

    def hs(self, tail=True, standard_name='sea_surface_wave_significant_height'):
        """
        Spectral significant wave height Hm0
        - tail  :: if True fit high-frequency tail
        - standard_name :: CF standard name for defining DataArray attribute
        """
        Sf = self.oned()
        E = 0.5 * (self.df * (Sf[{'freq': slice(1, None)}] + Sf[{'freq': slice(None, -1)}].values)).sum(dim='freq')
        if tail and Sf.freq[-1] > 0.333:
            E += 0.25 * Sf[{'freq': -1}].values * Sf.freq[-1].values
        hs = 4 * np.sqrt(E)
        hs.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'm'))))
        return hs.rename('hs')

    def tp(self, smooth=True, mask=np.nan, standard_name='sea_surface_wave_period_at_variance_spectral_density_maximum'):
        """
        Peak wave period
        - smooth :: if True returns the smooth wave period, if False simply returnsthe discrete
                    period corresponding to the maxima in the direction-integrated spectra
        - mask :: value for missing data in output (for when there is no peak in the frequency spectra)
        - standard_name :: CF standard name for defining DataArray attribute
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

        tp = (1+0*ipeak) / fp # Cheap way to turn Tp into appropriate DataArray object
        tp.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 's'))))

        # Returns masked dataarray
        return tp.where(ipeak>0).fillna(mask).rename('tp')

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

    def tm01(self, standard_name='sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment'):
        """
        Mean absolute wave period Tm01
        true average period from the 1st spectral moment
        - standard_name :: CF standard name for defining DataArray attribute
        """
        tm01 = self.momf(0).sum(dim='dir') / self.momf(1).sum(dim='dir')
        tm01.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 's'))))
        return tm01.rename('tm01')

    def tm02(self, standard_name='sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment'):
        """
        Mean absolute wave period Tm02
        Average period of zero up-crossings (Zhang, 2011)
        - standard_name :: CF standard name for defining DataArray attribute
        """
        tm02 = np.sqrt(self.momf(0).sum(dim='dir')/self.momf(2).sum(dim='dir'))
        tm02.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 's'))))
        return tm02.rename('tm02')

    def dm(self, standard_name='sea_surface_wave_from_direction'):
        """
        Mean wave direction from the 1st spectral moment
        - standard_name :: CF standard name for defining DataArray attribute
        """
        moms, momc = self.momd(1)
        dm = np.arctan2(moms.sum(dim='freq'), momc.sum(dim='freq'))
        dm = (270 - R2D*dm) % 360.
        dm.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'degree'))))
        return dm.rename('dm')

    def dp(self, standard_name='sea_surface_wave_direction_at_variance_spectral_density_maximum'):
        """
        Peak wave direction defined as the direction where the energy density
        of the frequency-integrated spectrum is maximum
        - standard_name :: CF standard name for defining DataArray attribute
        """
        ipeak = self._obj.sum(dim='freq').argmax(dim='dir')
        dp = self.dir.values[ipeak] * (ipeak*0+1) # Cheap way to turn dp into appropriate DataArray
        dp.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'degree'))))
        return dp.rename('dp')

    def dpm(self, mask=np.nan, standard_name='sea_surface_wave_peak_direction'):
        """
        From WW3 Manual: peak wave direction, defined like the mean direction, using the
        frequency/wavenumber bin containing of the spectrum F(k) that contains the peak frequency only
        - mask :: value for missing data in output (for when there is no peak in the frequency spectra)
        - standard_name :: CF standard name for defining DataArray attribute
        """
        ipeak = self._peak(self.oned())
        moms, momc = self.momd(1)
        moms_peak = self._collapse_array(moms.values, ipeak.values, axis=moms.get_axis_num(dim='freq'))
        momc_peak = self._collapse_array(momc.values, ipeak.values, axis=momc.get_axis_num(dim='freq'))
        dpm = np.arctan2(moms_peak, momc_peak) * (ipeak*0+1) # Cheap way to turn dp into appropriate DataArray
        dpm = (270 - R2D*dpm) % 360.
        dpm.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'degree'))))
        return dpm.where(ipeak>0).fillna(mask).rename('dpm')

    def dspr(self, standard_name='sea_surface_wave_directional_spread'):
        """
        Directional wave spreading, the one-sided directional width of the spectrum
        - standard_name :: CF standard name for defining DataArray attribute
        """
        moms, momc = self.momd(1)
        # Manipulate dimensions so calculations work
        moms = self._twod(moms, dim='dir')
        momc = self._twod(momc, dim='dir')
        mom0 = self._twod(self.momf(0), dim='freq').spec.oned()

        dspr = (2 * R2D**2 * (1 - ((moms.spec.momf()**2 + momc.spec.momf()**2)**0.5 / mom0)))**0.5
        dspr = dspr.sum(dim='freq').sum(dim='dir')
        dspr.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'degree'))))
        return dspr.rename('dspr')

    def swe(self, standard_name='sea_surface_wave_spectral_width'):
        """
        Spectral width parameter by Cartwright and Longuet-Higgins (1956)
        Represents the range of frequencies where the dominant energy exists
        - standard_name :: CF standard name for defining DataArray attribute
        """
        swe = (1. - self.momf(2).sum(dim='dir')**2 / (self.momf(0).sum(dim='dir')*self.momf(4).sum(dim='dir')))**0.5
        swe.values[swe.values < 0.001] = 1.
        swe.attrs.update(OrderedDict((('standard_name', standard_name), ('units', ''))))
        return swe.rename('swe')

    def sw(self, mask=np.nan, standard_name='sea_surface_wave_spectral_width'):
        """
        Spectral width parameter by Longuet-Higgins (1975)
        Represents energy distribution over entire frequency range
        - mask :: value for missing data in output (for when Hs is smaller then 0.001 m)
        - standard_name :: CF standard name for defining DataArray attribute
        """
        sw = (self.momf(0).sum(dim='dir') * self.momf(2).sum(dim='dir') / self.momf(1).sum(dim='dir')**2 - 1.0)**0.5
        sw.attrs.update(OrderedDict((('standard_name', standard_name), ('units', ''))))
        return sw.where(self.hs() >= 0.001).fillna(mask).rename('sw')

    # def partition(self, wsp, wdir, dep, agefac=1.7, wscut=0.3333):
    #     """
    #     Watershed partition
    #     - wsp :: Wind speed
    #     - wdir :: Wind direction in coming from convention
    #     - dep :: Water depth
    #     """
    #     import pymo.core.specpart
    #     ipart = pymo.core.specpart.specpart.partition(self.S)
    #     npart = ipart.max()
    #     sea = Spectrum(self.freqs, self.dirs)
    #     swell = []
    #     for np in range(1, npart+1):
    #         stmp = Spectrum(self.freqs, self.dirs, numpy.where(ipart==np, self.S,0.))
    #         imax, imin = stmp.shape(0.01,0.05);
    #         if len(imin) > 0:
    #             stmp.S[imin[0]:,:] = 0
    #             newpart = (stmp.S>0)
    #             if newpart.sum() > 20:
    #                 npart = npart+1;
    #                 ipart[newpart] = npart
    #     for np in range(1, npart+1):
    #         stmp = Spectrum(self.freqs, self.dirs, numpy.where(ipart==np,s elf.S, 0.))
    #         Up = agefac * wsp * numpy.cos(d2r*(self.dirs-wdir))
    #         W = stmp.S[numpy.tile(Up,(len(self.freqs),1)) > numpy.tile(self.C(dep)[:,numpy.newaxis],(1,len(self.dirs)))].sum()/stmp.S.sum()
    #         if W > wscut:
    #             sea += stmp
    #         else:
    #             stmp._hsig = stmp.hs()
    #             if stmp._hsig > 0.001:
    #                 swell.append(stmp)
    #         if len(swell) > 1:
    #             swell.sort(key=lambda spec: spec._hsig, reverse=True)
    #     return [sea] + swell

#Add all the export functions at class creation time
#Also add wrapper functions for all the SpecArray methods
class DatasetPlugin(type):
    def __new__(cls,name,bases,dct):
        import io
        for fname in dir(io):
            if 'to_' not in fname:continue
            function = getattr(io,fname)
            if isinstance(function, types.FunctionType):
                dct[function.__name__] = function
        return type.__new__(cls, name, bases, dct)
        

#This just provides a wrapper around the xarray dataset 
class SpecDataset(object):
    __metaclass__ = DatasetPlugin
    def __init__(self, xarray_dset):
        self.dset=xarray_dset
        
    def __repr__(self):
        return 'Spectral Dataset wrapper'+str(self.dset)
    
    def __getattr__(self,fn):
        if fn in dir(SpecArray) and (fn[0]!='_'):
            return getattr(self.dset['efth'].spec,fn)
        else:
            return getattr(self.dset,fn)
    
            

if __name__ == '__main__':
    pass