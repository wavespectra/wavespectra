"""
Spectra object based on DataArray with methods to calculate spectral statistics

Reference:
- Cartwright and Longuet-Higgins (1956). The Statistical Distribution of the Maxima of a Random Function,
        Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, 237, 212-232.
- Longuet-Higgins (1975). On the joint distribution of the periods and amplitudes of sea waves,
        Journal of Geophysical Research, 80, 2688-2694, doi: 10.1029/JC080i018p02688.
- Zhang (2011).
"""
import re
from collections import OrderedDict
from datetime import datetime
import numpy as np
import xarray as xr
import types
import copy
from itertools import product
from sympy import Symbol
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr

from misc import GAMMA, D2R, R2D

# TODO: dimension renaming and sorting in __init__ are not producing intended effect. They correctly modify xarray_obj
#       as defined in xarray.spec._obj but the actual xarray is not modified - and they loose their direct association
# TODO: Implement true_peak method for both tp() and dpm()

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
        self._non_spec_dims = set(self._obj.dims).difference(('freq', 'dir'))

        # Create attributes for accessor 
        self.freq = xarray_obj.freq
        self.dir = xarray_obj.dir if 'dir' in xarray_obj.dims else None

        self.df = abs(self.freq[1:].values - self.freq[:-1].values) if len(self.freq) > 1 else np.array((1.0,))
        self.dd = abs(self.dir[1].values - self.dir[0].values) if self.dir is not None and len(self.dir) > 1 else 1.0

        # df darray with freq dimension - may replace above one in the future
        if len(self.freq) > 1:
            self.dfarr = xr.DataArray(data=np.hstack((1.0, np.full(len(self.freq)-2, 0.5), 1.0)) *\
                                           (np.hstack((0.0, np.diff(self.freq))) + np.hstack((np.diff(self.freq), 0.0))),
                                      coords={'freq': self.freq},
                                      dims=('freq'))
        else:
            self.dfarr  = xr.DataArray(data=np.array((1.0,)),
                                       coords={'freq': self.freq},
                                       dims=('freq'))

    def __repr__(self):
        return re.sub(r'<([^\s]+)', '<%s'%(self.__class__.__name__), str(self._obj))

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

    def _product(self, dict_of_ids):
        """
        Dot product of a dictionary of ids used to construct arbitrary slicing dictionaries
        e.g.:
            Input dictionary :: {'site': [0,1], 'time': [0,1]}
            Output iterator :: {'site': 0, 'time': 0}
                               {'site': 0, 'time': 1}
                               {'site': 1, 'time': 0}
                               {'site': 1, 'time': 1}
        """
        return (dict(zip(dict_of_ids, x)) for x in product(*dict_of_ids.itervalues()))

    def _inflection(self, fdspec, dfres=0.01, fmin=0.05):
        """
        Finds points of inflection in smoothed frequency spectra
        - fdspec :: freq-dir 2D specarray
        - dfres :: used to determine length of smoothing window
        - fmin :: minimum frequency for looking for minima/maxima
        """
        if len(self.freq) > 1:
            sf = fdspec.sum(axis=1)
            nsmooth = int(dfres / self.df[0]) # Window size
            if nsmooth > 1:
                sf = np.convolve(sf, np.hamming(nsmooth), 'same') # Smoothed f-spectrum
            sf[(sf<0) | (self.freq.values<fmin)] = 0
            diff = np.diff(sf)
            imax = np.argwhere(np.diff(np.sign(diff)) == -2) + 1
            imin = np.argwhere(np.diff(np.sign(diff)) == 2) + 1
        else:
            imax = 0
            imin = 0
        return imax, imin

    def _same_dims(self, other):
        """
        True if other has same non-spectral dimensions
        Used to ensure consistent slicing
        """
        return set(other.dims) == self._non_spec_dims

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

    def oned(self, skipna=True):
        """
        Returns the one-dimensional frequency spectra
        Direction dimension is dropped after integrating
        - skipna :: if True (xarray's default) sum of all-nan array returns zero preventing hs() to preserve mask 
        """
        if self.dir is not None:
            return self.dd * self._obj.sum(dim='dir', skipna=skipna)
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
        if 'dir' in other.dims and (dmin or dmax):
            other = self.sort(other, dims=['dir']).sel(dir=slice(dmin, dmax))

        # Interpolate at fmin
        if (other.freq.min() > fmin) and (self.freq.min() <= fmin):
            other = xr.concat([self._interp_freq(fmin), other], dim='freq')

        # Interpolate at fmax
        if (other.freq.max() < fmax) and (self.freq.max() >= fmax):
            other = xr.concat([other, self._interp_freq(fmax)], dim='freq') 

        return other

    def to_energy(self, standard_name='sea_surface_wave_directional_energy_spectra'):
        """
        Convert from energy density (m2/Hz/degree) into wave energy spectra (m2)
        """
        E = self._obj * self.dfarr * self.dd
        E.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'm^{2}'))))
        return E.rename('energy')

    def hs(self, tail=True, standard_name='sea_surface_wave_significant_height'):
        """
        Spectral significant wave height Hm0
        - tail  :: if True fit high-frequency tail
        - standard_name :: CF standard name for defining DataArray attribute
        """
        Sf = self.oned(skipna=False)
        E = (Sf * self.dfarr).sum(dim='freq')
        if tail and Sf.freq[-1] > 0.333:
            E += 0.25 * Sf[{'freq': -1}] * Sf.freq[-1].values
        hs = 4 * np.sqrt(E)
        hs.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'm'))))
        return hs.rename('hs')
    
    def scale_by_hs(self, expr, inplace=True, hs_min=-np.inf, hs_max=np.inf,
                    tp_min=-np.inf, tp_max=np.inf, dpm_min=-np.inf, dpm_max=np.inf):
        """
        Correct spectra using equation based on hs
        - expr :: str - expression to apply, e.g. '0.13*hs + 0.02'
        - inplace :: bool - use True to apply transformation in place, False otherwise
        - hs_min, hs_max :: float or inf specifying Hs range over which expr is applied
        - tp_min, tp_max :: float or inf specifying Tp range over which expr is applied
        - dpm_min, dpm_max :: float or inf specifying Dpm range over which expr is applied
        """
        func = lambdify(Symbol('hs'), parse_expr(expr.lower()))
        hs = self.hs()
        k = (func(hs) / hs)**2
        scaled = k * self._obj
        if any([abs(arg) != np.inf for arg in [hs_min, hs_max, tp_min, tp_max, dpm_min, dpm_max]]):
            tp = self.tp()
            dpm = self.dpm()
            scaled = scaled.where(( (hs>=hs_min) & (hs<=hs_max) &
                                    (tp>=tp_min) & (tp<=tp_max) & 
                                    (dpm>=dpm_min) & (dpm<=dpm_max)
                                   )).combine_first(self._obj)
        if inplace:
            self._obj.values = scaled.values
        else:
            return scaled

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
        fp = self.freq**mom
        mf = self.dfarr * fp * self._obj
        return self._twod(mf.sum(dim='freq', skipna=False)).rename('mom{:d}'.format(mom))

    def momd(self, mom=0, theta=90.):
        """
        Directional moment
        """
        cp = np.cos(np.radians(180 + theta - self.dir))**mom
        sp = np.sin(np.radians(180 + theta - self.dir))**mom
        msin = (self.dd * self._obj * sp).sum(dim='dir', skipna=False)
        mcos = (self.dd * self._obj * cp).sum(dim='dir', skipna=False)
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

    def dm(self, standard_name='sea_surface_wave_from_mean_direction'):
        """
        Mean wave direction from the 1st spectral moment
        - standard_name :: CF standard name for defining DataArray attribute
        """
        moms, momc = self.momd(1)
        dm = np.arctan2(moms.sum(dim='freq', skipna=False), momc.sum(dim='freq', skipna=False))
        dm = (270 - R2D*dm) % 360.
        dm.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'degree'))))
        return dm.rename('dm')

    def dp(self, standard_name='sea_surface_wave_from_direction_at_variance_spectral_density_maximum'):
        """
        Peak wave direction defined as the direction where the energy density
        of the frequency-integrated spectrum is maximum
        - standard_name :: CF standard name for defining DataArray attribute
        """
        ipeak = self._obj.sum(dim='freq').argmax(dim='dir')
        template = self._obj.sum(dim='freq', skipna=False).sum(dim='dir', skipna=False)
        dp = self.dir.values[ipeak.values] * (0 * template + 1)
        dp.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'degree'))))
        return dp.rename('dp')

    def dpm(self, mask=np.nan, standard_name='sea_surface_wave_from_direction_at_variance_spectral_density_maximum'):
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
        mom0 = self._twod(self.momf(0), dim='freq').spec.oned(skipna=False)

        dspr = (2 * R2D**2 * (1 - ((moms.spec.momf()**2 + momc.spec.momf()**2)**0.5 / mom0)))**0.5
        dspr = dspr.sum(dim='freq', skipna=False).sum(dim='dir', skipna=False)
        dspr.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'degree'))))
        return dspr.rename('dspr')

    def crsd(self, theta=90., standard_name='sea_surface_wave_update_missing_part_here'):
        """
        Add description for this method
        """
        cp = np.cos(D2R * (180 + theta - self.dir))
        sp = np.sin(D2R * (180 + theta - self.dir))
        crsd = (self.dd * self._obj * cp * sp).sum(dim='dir')
        crsd.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'm2s'))))
        return crsd.rename('crsd')

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

    def celerity(self, depth=False, freq=None):
        """
        Wave celerity
        Deep water approximation is returned if depth=False
        """
        if freq is None:
            freq = self.freq
        if depth:
            ang_freq = 2 * np.pi * freq
            return ang_freq / wavenuma(ang_freq, depth)
        else:
            return 1.56 / freq

    def wavelen(self, depth=False):
        """
        Wave length
        Deep water approximation is returned if depth=False
        """
        if depth:
            ang_freq = 2 * np.pi * self.freq
            return 2 * np.pi / wavenuma(ang_freq, depth)
        else:
            return 1.56 / self.freq**2

    def partition(self, wsp_darr, wdir_darr, dep_darr, agefac=1.7, wscut=0.3333, hs_min=0.001, nearest=False):
        """
        Partition wave spectra using WW3 watershed algorithm
        Input:
            - wsp_darr :: Wind speed DataArray (m/s) - must have same non-spectral dimensions as specarray
            - wdir_darr :: Wind direction DataArray (degree) - must have same non-spectral dimensions as specarray
            - dep_darr :: Water depth DataArray(m) - must have same non-spectral dimensions as specarray
            - hs_min :: minimum Hs for assigning swell partition
            - nearest :: if True, wsp wdir and dep are allowed to be taken from nearest point (slower)
        Output:
            - part_spec :: SpecArray object with one extra dimension representig partition number
        """

        # Assert spectral dims are present in spectra and non-spectral dims are present in winds and depths
        assert 'freq' in self._obj.dims and 'dir' in self._obj.dims, ('partition requires E(freq,dir) but freq|dir '
            'dimensions not in SpecArray dimensions (%s)' % (self._obj.dims))
        for darr in (wsp_darr, wdir_darr, dep_darr):
            # Conditional below aims at allowing wsp, wdir, dep to be DataArrays within the SpecArray. not working yet
            if isinstance(darr, str):
                darr = getattr(self, darr)
            assert set(darr.dims)==self._non_spec_dims, ('%s dimensions (%s) need matching non-spectral dimensions '
                'in SpecArray (%s) for consistent slicing' % (darr.name, set(darr.dims), self._non_spec_dims))

        from spectra.specpart import specpart

        # Initialise output - one SpecArray for each partition
        all_parts = [0 * self._obj]

        # Predefine these for speed
        dirs = self.dir.values
        freqs = self.freq.values
        ndir = len(dirs)
        nfreq = len(freqs)

        # Slice each possible 2D, freq-dir array out of the full data array
        slice_ids = {dim: range(self._obj[dim].size) for dim in self._non_spec_dims}
        for slice_dict in self._product(slice_ids):
            specarr = self._obj[slice_dict]
            if nearest:
                slice_dict_nearest = {key: self._obj[key][val].values for key,val in slice_dict.items()}
                wsp = float(wsp_darr.sel(method='nearest', **slice_dict_nearest))
                wdir = float(wdir_darr.sel(method='nearest', **slice_dict_nearest))
                dep = float(dep_darr.sel(method='nearest', **slice_dict_nearest))
            else:
                wsp = float(wsp_darr[slice_dict])
                wdir = float(wdir_darr[slice_dict])
                dep = float(dep_darr[slice_dict])

            spectrum = specarr.values
            part_array = specpart.partition(spectrum)
            nparts = part_array.max()

            # Assign new partition if multiple valleys and satisfying some conditions
            for part in range(1, nparts+1):
                part_spec = np.where(part_array==part, spectrum, 0.) # Current partition only
                imax, imin = self._inflection(part_spec, dfres=0.01, fmin=0.05)
                if len(imin) > 0:
                    part_spec[imin[0].squeeze():, :] = 0
                    newpart = part_spec > 0
                    if newpart.sum() > 20:
                        nparts += 1
                        part_array[newpart] = nparts
                
            # Extend partitions list if any extra one has been detected (+1 because of sea)
            if len(all_parts) < nparts+1:
                for new_part_number in set(range(nparts+1)).difference(range(len(all_parts))):
                    all_parts.append(0 * self._obj)

            # Assign sea and swells partitions based on wind and wave properties
            sea = 0 * spectrum
            swells = list()
            hs_swell = list()
            Up = agefac * wsp * np.cos(D2R*(dirs - wdir))
            for part in range(1, nparts+1):
                part_spec = np.where(part_array==part, spectrum, 0.) # Current partition only
                W = part_spec[np.tile(Up, (nfreq, 1)) > \
                    np.tile(self.celerity(dep, freqs)[:,_], (1, ndir))].sum() / part_spec.sum()
                if W > wscut:
                    sea += part_spec
                else:
                    _hs = hs(part_spec, freqs, dirs)
                    if _hs > hs_min:
                        swells.append(part_spec)
                        hs_swell.append(_hs)
                        hs_swell = [h + np.random.rand()*1e-10 for h in hs_swell] # If two or more partitions with same Hs
            if len(swells) > 1:
                swells = [x for (y,x) in sorted(zip(hs_swell, swells), reverse=True)]

            # Updating partition SpecArrays for current slice
            all_parts[0][slice_dict] = sea
            for ind, swell in enumerate(swells):
                all_parts[ind+1][slice_dict] = swell

        # Concatenate partitions along new axis
        part_coord = xr.DataArray(data=range(len(all_parts)),
                                  coords={'part': range(len(all_parts))},
                                  dims=('part',),
                                  name='part',
                                  attrs=OrderedDict((('standard_name', 'spectral_partition_number'), ('units', '')))
                                  )
        return xr.concat(all_parts, dim=part_coord)

    def dpw(self):
        """
        Wave spreading at the peak wave frequency
        """
        raise NotImplementedError('Wave spreading at the peak wave frequency method not defined')

    def stats(self, stats, fmin=None, fmax=None, dmin=None, dmax=None, names=None):
        """
        Calculate multiple spectral stats into one same DataArray
        Input:
            - stats :: list of strings specifying stats to be calculated, or alternatively dictionary where
                       keys are stats and values are dictionaries with kwargs to use with corresponding method
            - fmin, fmax :: float, lower and upper frequencies for splitting spectra before calculating stats
            - dmin, dmax :: float, lower and upper directions for splitting spectra before calculating stats
            - names :: list of strings, used to rename each stat in output Dataset
        Returns:
            - DataArray if only one stat, Dataset otherwise
        Remarks:
            - all stats must correspond to implemented methods in this class
            - if names is provided, its length must correspond to the length of stats
        """
        if any((fmin, fmax, dmin, dmax)):
            spectra = self.split(fmin=fmin, fmax=fmax, dmin=dmin, dmax=dmax)
        else:
            spectra = self._obj
        
        if isinstance(stats, (list, tuple)):
            stats_dict = {s: {} for s in stats}
        elif isinstance(stats, dict):
            stats_dict = stats
        else:
            raise ValueError('stats must be either a container or a dictionary')
        
        names = names or stats_dict.keys()
        if len(names) != len(stats_dict):
            raise ValueError('length of names does not correspond to the number of stats')

        params = list()
        for name, (func, kwargs) in zip(names, stats_dict.items()):
            try:
                stats_func = getattr(spectra.spec, func)
            except:
                raise IOError('%s is not implemented as a method in %s' % (func, self.__class__.__name__))
            if callable(stats_func):
                params.append(stats_func(**kwargs).rename(name))
            else:
                raise IOError('%s attribute of %s is not callable' % (func, self.__class__.__name__))

        return xr.merge(params)


def wavenuma(ang_freq, water_depth):
    """
    Chen and Thomson wavenumber approximation
    """
    k0h = 0.10194 * ang_freq * ang_freq * water_depth
    D = [0,0.6522,0.4622,0,0.0864,0.0675]
    a = 1.0
    for i in range(1, 6):
        a += D[i] * k0h**i
    return (k0h * (1 + 1./(k0h*a))**0.5) / water_depth

def hs(spec, freqs, dirs, tail=True):
    """
    Copied as a function so it can be used in a generic context
    - tail  :: if True fit high-frequency tail
    """
    df = abs(freqs[1:] - freqs[:-1])
    if len(dirs) > 1:
        ddir = abs(dirs[1] - dirs[0])
        E = ddir * spec.sum(1)
    else:
        E = np.squeeze(spec)
    Etot = 0.5 * sum(df * (E[1:] + E[:-1]))
    if tail and freqs[-1] > 0.333:
        Etot += 0.25 * E[-1] * freqs[-1]
    return 4. * np.sqrt(Etot)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from spectra.readspec import read_swan

    ds1 = read_swan('/source/pyspectra/tests/antf0.20170207_06z.bnd.swn')
    ds2 = ds1.copy(deep=True)
    ds2.spec.scale_by_hs('2*hs', inplace=True)
    hs1 = ds1.spec.hs().values.ravel()
    hs2 = ds2.spec.hs().values.ravel()
    fig = plt.figure()
    plt.plot(hs1, color='k')
    plt.plot(hs2, color='b')
    plt.show()

    # from spectra.readspec import read_hotswan
    # hot = read_hotswan('/source/pyspectra/tests/swan/hot/aklishr*.hot*')
    # import datetime
    # import matplotlib.pyplot as plt
    # from os.path import expanduser, join
    # home = expanduser("~")

    # filename = '/source/pyspectra/tests/prelud.spec'
    # # filename = '/source/pyspectra/tests/antf0.20170207_06z.bnd.swn'

    # wsp_val = 10
    # wdir_val = 225
    # dep_val = 100
    # #================
    # # Using SpecArray
    # #================
    # from readspec import read_swan
    # t0 = datetime.datetime.now()
    # ds = read_swan(filename, dirorder=True)
    # # Fake wsp, wdir, dep
    # wsp  = ds.hs() * 0 + wsp_val
    # wdir = ds.hs() * 0 + wdir_val
    # dep  = ds.hs() * 0 + dep_val
    # parts = ds.partition(wsp, wdir, dep)
    # # ds.efth['wsp'] = wsp
    # # ds.efth['wdir'] = wdir
    # # ds.efth['dep'] = dep
    # # parts = ds.partition('wsp', 'wdir', 'dep')
    # print 'Elapsed time new: %0.2f s' % ((datetime.datetime.now() - t0).total_seconds())

    # #================
    # # Using pymo
    # #================
    # from pymo.data.spectra import SwanSpecFile
    # t0 = datetime.datetime.now()
    # specfile = SwanSpecFile(filename)
    # spectra = [spec for spec in specfile.readall()]
    # pymo_parts = [s.partition(wsp_val, wdir_val, dep_val) for s in spectra]
    # print 'Elapsed time old: %0.2f s' % ((datetime.datetime.now() - t0).total_seconds())

    # for p in range(0,3):
    #     hs_new = parts.isel(part=p, lat=0, lon=0).spec.hs()
    #     hs_old = [part[p].hs() for part in pymo_parts]

    #     plt.figure()
    #     hs_new.plot()
    #     plt.plot(hs_new.time, hs_old)
    #     plt.legend(('SpecArray', 'Pymo'))
    #     plt.savefig(join(home,'Pictures/compare_hs_part%i.png' % (p)))

    #     # break
