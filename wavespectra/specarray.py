"""Spectra object based on DataArray to calculate spectral statistics.

Reference:
    - Cartwright and Longuet-Higgins (1956). The Statistical Distribution of the Maxima
      of a Random Function, Proceedings of the Royal Society of London. Series A,
      Mathematical and Physical Sciences, 237, 212-232.
    - Holthuijsen LH (2005). Waves in oceanic and coastal waters (page 82).
    - Longuet-Higgins (1975). On the joint distribution of the periods and amplitudes
      of sea waves, Journal of Geophysical Research, 80, 2688-2694.

"""
import re
import numpy as np
import xarray as xr
from itertools import product
import inspect
import warnings

from wavespectra.plot import _PlotMethods
from wavespectra.core.attributes import attrs
from wavespectra.core.utils import D2R, R2D, celerity, wavenuma, wavelen, bins_from_frequency_grid
from wavespectra.core.watershed import partition
from wavespectra.core import xrstats


# TODO: dimension renaming and sorting in __init__ are not producing intended effect.
#       They correctly modify xarray_obj as defined in xarray.spec._obj but the actual
#       xarray is not modified - and they loose their direct association


@xr.register_dataarray_accessor("spec")
class SpecArray(object):
    """Extends DataArray with methods to deal with wave spectra."""

    def __init__(self, xarray_obj):
        """Initialise spec accessor."""
        self._obj = xarray_obj

        # These are set when property is first called to avoid computing more than once
        self._df = None
        self._dd = None
        self._dfarr = None

    def __repr__(self):
        return re.sub(r"<([^\s]+)", "<%s" % (self.__class__.__name__), str(self._obj))

    @property
    def freq(self):
        """Frequency DataArray."""
        return self._obj.freq

    @property
    def dir(self):
        """Direction DataArray."""
        if attrs.DIRNAME in self._obj.dims:
            return self._obj[attrs.DIRNAME]
        else:
            return None

    @property
    def _non_spec_dims(self):
        """Return the set of non-spectral dimensions in underlying dataset."""
        return set(self._obj.dims).difference((attrs.FREQNAME, attrs.DIRNAME))

    @property
    def dfarr(self):
        """Frequency resolution DataArray."""
        if self._dfarr is not None:
            return self._dfarr
        if len(self.freq) > 1:
            fact = np.hstack((1.0, np.full(self.freq.size - 2, 0.5), 1.0))
            ldif = np.hstack((0.0, np.diff(self.freq)))
            rdif = np.hstack((np.diff(self.freq), 0.0))
            self._dfarr = xr.DataArray(data=fact * (ldif + rdif), coords=self.freq.coords)
        else:
            self._dfarr = xr.DataArray( data=np.array((1.0,)), coords=self.freq.coords)
        return self._dfarr

    @property
    def df(self):
        """Frequency resolution numpy array.

        TODO: Check if this can be removed in favor of dfarr.

        """
        if self._df is not None:
            return self._df
        if len(self.freq) > 1:
            self._df = abs(self.freq[1:].values - self.freq[:-1].values)
        else:
            self._df = np.array((1.0,))
        return self._df

    @property
    def dd(self):
        """Direction resolution float."""
        if self._dd is not None:
            return self._dd
        if self.dir is not None and len(self.dir) > 1:
            self._dd = abs(float(self.dir[1] - self.dir[0]))
        else:
            self._dd = 1.0
        return self._dd

    def _strictly_increasing(self, arr):
        """Check if array is sorted in increasing order."""
        return all(x < y for x, y in zip(arr, arr[1:]))

    def _collapse_array(self, arr, indices, axis):
        """Collapse ndim array [arr] along [axis] using [indices]."""
        magic_index = [np.arange(i) for i in indices.shape]
        magic_index = np.ix_(*magic_index)
        magic_index = magic_index[:axis] + (indices,) + magic_index[axis:]
        return arr[magic_index]

    def _twod(self, darray, dim=attrs.DIRNAME):
        """Ensure dir,freq dims are present so moment calculations won't break."""
        if dim not in darray.dims:
            return darray.expand_dims(dim={dim: [1]}, axis=-1)
        else:
            return darray

    def _interp_freq(self, fint):
        """Linearly interpolate spectra at frequency fint.

        Assumes self.freq.min() < fint < self.freq.max()

        Returns:
            DataArray with one value in frequency dimension (relative to fint)
            otherwise same dimensions as self._obj

        """
        if not (self.freq.min() < fint < self.freq.max()):
            raise ValueError(
                f"fint must be within freq range {self.freq.values}, got {fint}"
            )
        ifreq = self.freq.searchsorted(fint)
        df = np.diff(self.freq.isel(freq=[ifreq - 1, ifreq]))[0]

        right = self._obj.isel(freq=[ifreq]) * (fint - self.freq[ifreq - 1])
        left = self._obj.isel(freq=[ifreq - 1]) * (self.freq[ifreq] - fint)

        right = right.assign_coords({"freq": [fint]})
        left = left.assign_coords({"freq": [fint]})

        return (left + right) / df

    def _peak(self, arr):
        """Returns indices of largest peaks along freq dim in a ND-array.

        Args:
            - arr (SpecArray): 1D spectra (integrated over directions)

        Returns:
            - ipeak (SpecArray): indices for slicing arr at the frequency peak

        Note:
            - Peak is defined IFF arr(ipeak-1) < arr(ipeak) < arr(ipeak+1).
            - Values at the array boundary do not satisfy above condition and treated
              as missing_value in other parts of the code.
            - Flat peaks are ignored by this criterium.

        """
        fwd = xr.concat((arr.isel(freq=0), arr), dim=attrs.FREQNAME).diff(
            attrs.FREQNAME, n=1, label="upper"
        ) > 0
        bwd = xr.concat((arr, arr.isel(freq=-1)), dim=attrs.FREQNAME).diff(
                attrs.FREQNAME, n=1, label="lower"
        ) < 0
        ispeak = np.logical_and(fwd, bwd)
        return arr.where(ispeak, 0).argmax(dim=attrs.FREQNAME).astype(int)

    def _my_name(self):
        """Returns the caller's name."""
        return inspect.stack()[1][3]

    def _standard_name(self, varname):
        try:
            return attrs.ATTRS[varname]["standard_name"]
        except AttributeError:
            warnings.warn(
                f"Cannot set standard_name for variable {varname}. "
                "Ensure it is defined in attributes.yml"
            )
            return ""

    def _units(self, varname):
        try:
            return attrs.ATTRS[varname]["units"]
        except AttributeError:
            warnings.warn(
                f"Cannot set units for variable {varname}. "
                "Ensure it is defined in attributes.yml"
            )
            return ""

    @property
    def plot(self) -> _PlotMethods:
        """Access plotting functions.

        For convenience just call this directly:

            * dset.efth.spec.plot()
            * dset.spec.plot()

        Or use it as a namespace to use wavespectra.plot functions as:

            * dset.efth.spec.plot.contourf()
            * dset.spec.plot.contourf()

        Which is equivalent to wavespectra.plot.imshow(dset.efth)

        """
        return _PlotMethods(self._obj)

    def oned(self, skipna=True):
        """Returns the one-dimensional frequency spectra.

        Direction dimension is dropped after integrating.

        Args:
            - skipna (bool): choose it to skip nans when integrating spectra.
              This is the default behaviour for sum() in DataArray. Notice it
              converts masks, where the entire array is nan, into zero.

        """
        if self.dir is not None:
            return self.dd * self._obj.sum(dim=attrs.DIRNAME, skipna=skipna)
        else:
            return self._obj.copy(deep=True)

    def split(self, fmin=None, fmax=None, dmin=None, dmax=None, rechunk=True):
        """Split spectra over freq and/or dir dims.

        Args:
            - fmin (float): lowest frequency to split spectra, by default the lowest.
            - fmax (float): highest frequency to split spectra, by default the highest.
            - dmin (float): lowest direction to split spectra at, by default min(dir).
            - dmax (float): highest direction to split spectra at, by default max(dir).
            - rechunk (bool): Rechunk split dims so there is one single chunk.

        Note:
            - Spectra are interpolated at `fmin` / `fmax` if they are not in self.freq.
            - Recommended rechunk==True so ufuncs with freq/dir as core dims will work.

        """
        if fmax is not None and fmin is not None and fmax <= fmin:
            raise ValueError("fmax needs to be greater than fmin")
        if dmax is not None and dmin is not None and dmax <= dmin:
            raise ValueError("dmax needs to be greater than dmin")

        # Slice frequencies
        other = self._obj.sel(freq=slice(fmin, fmax))

        # Slice directions
        if attrs.DIRNAME in other.dims and (dmin or dmax):
            other = self._obj.sortby([attrs.DIRNAME]).sel(dir=slice(dmin, dmax))

        # Interpolate at fmin
        if fmin is not None and (other.freq.min() > fmin) and (self.freq.min() <= fmin):
            other = xr.concat([self._interp_freq(fmin), other], dim=attrs.FREQNAME)

        # Interpolate at fmax
        if fmax is not None and (other.freq.max() < fmax) and (self.freq.max() >= fmax):
            other = xr.concat([other, self._interp_freq(fmax)], dim=attrs.FREQNAME)

        other.freq.attrs = self._obj.freq.attrs
        other.dir.attrs = self._obj.dir.attrs

        if rechunk:
            other = other.chunk({attrs.FREQNAME: None, attrs.DIRNAME: None})

        return other

    def to_energy(self, standard_name="sea_surface_wave_directional_energy_spectra"):
        """Convert from energy density (m2/Hz/degree) into wave energy spectra (m2)."""
        E = self._obj * self.dfarr * self.dd
        E.attrs.update({"standard_name": standard_name, "units": "m^{2}"})
        return E.rename("energy")

    def from_bins_to_continuous(self):
        """Assuming that the spectral data currently in the array represents the spectrum by bins,
        converts this to an equivalent continuous spectral density.

        See:
            documentation : spectral_energy_and_bins
        """

        efth = self._obj # alias
        # Based on the grid, try to obtain the bin edges

        left, right, width, center = bins_from_frequency_grid(self.freq)

        m0_bins = np.sum(width * efth.data)

        update_rate = 0.5  # new iteration = update_rate *guess + (1-update_rate) * actual_values
        last_max_update = 0

        # the spectal efth data is lcated at self._obj
        #
        # get the dim of the freq axis


        efth.dims

        # s_datapoint = self.spec.efth.values

        last_max_update = 0

        f_left = center[:-1]  # datapoint left of current
        f_right = center[1:]  # datapoint right of current
        edge_internal = left[1:]
        dfl = edge_internal - f_left
        dfr = f_right - edge_internal

        e0 = efth.data * width

        for i in range(100):

            s_left = efth.data[:-1]
            s_right = efth.data[1:]



            # s_edge_internal = (dfl * s_left + dfr * s_right) / (dfr + dfl)
            s_edge_internal = s_left + dfl * (s_right - s_left) / (dfr + dfl)

            # values at the outer edges are zero
            s_edge = np.array([0, *s_edge_internal, 0], dtype=float)

            # move the data-points such that the energy per bins stays constant
            #
            # Energy in an original bin = bin_value * widths
            # Energy in the continuous spectrum in the same area as the bin
            #
            # the spectral density at the edges is s_edge
            # the spectral density at the locations of the original datapoint can now be calculated
            #
            # bin_value * width =
            # 0.5 * (s_left_edge + s_datapoint) * (datapoint - left_edge) +
            # 0.5 * (s_right_edge + s_datapoint) * (right_edge - datapoint) +

            d_left = center - left
            d_right = right - center

            s_left = s_edge[:-1]
            s_right = s_edge[1:]

            # e0 =
            # 0.5 * (s_left_edge + s_datapoint) * d_left +
            # 0.5 * (s_right_edge + s_datapoint) * d_right
            # =
            # 0.5*s_left_edge * d_left + 0.5*s_datapoint * d_left + 0.5 * s_right*edge*d_right + 0.5 * s_datapoint * d_right
            #
            # 0.5 * s_datapoint ( d_left + d_right ) = e0 - 0.5 * s_left_edge * d_left - 0.5 * s_right_edge * d_right

            new_estimate = (e0 - 0.5 * s_left * d_left - 0.5 * s_right * d_right) / (0.5 * (d_left + d_right))

            # we may have gotten datapoints below zero
            new_estimate = np.maximum(new_estimate, 0)  # note, not max!

            i_maxchange = np.argmax(np.abs(new_estimate - efth.data))
            max_update = (new_estimate -  efth.data)[i_maxchange]  # signed

            change_in_update = max_update - last_max_update
            last_max_update = max_update
            print(f'change in update: {change_in_update}')

            if abs(change_in_update) < 1e-5:
                break

            print(f'it {i} , max update {max_update} as point {i_maxchange}')

            efth.data = update_rate * new_estimate + (1 - update_rate) * efth.data

            # plt.plot(datapoints, s_datapoint, 'g*')

            # scale to original m0
            m0_cont = -np.trapz(center, efth.data)
            scale = m0_bins / m0_cont

            efth.data *= scale

        print('done')



    def hs(self, tail=True):
        """Spectral significant wave height Hm0.

        Args:
            - tail (bool): if True fit high-frequency tail before integrating spectra.

        """
        Sf = self.oned(skipna=False)
        E = (Sf * self.dfarr).sum(dim=attrs.FREQNAME)
        if tail and Sf.freq[-1] > 0.333:
            E += (
                0.25
                * Sf[{attrs.FREQNAME: -1}].drop_vars(attrs.FREQNAME)
                * Sf.freq[-1].values
            )
        hs = 4 * np.sqrt(E)
        hs.attrs.update(
            {
                "standard_name": self._standard_name(self._my_name()),
                "units": self._units(self._my_name()),
            }
        )
        return hs.rename(self._my_name())

    def hmax(self):
        """Maximum wave height Hmax.

        hmax is the most probably value of the maximum individual wave height
        for each sea state. Note that maximum wave height can be higher (but
        not by much since the probability density function is rather narrow).

        Reference:
            - Holthuijsen LH (2005). Waves in oceanic and coastal waters (page 82).

        """
        if attrs.TIMENAME in self._obj.coords and self._obj.time.size > 1:
            dt = np.diff(self._obj.time).astype("timedelta64[s]").mean()
            N = (
                dt.astype(float) / self.tm02()
            ).round()  # N is the number of waves in a sea state
            k = np.sqrt(0.5 * np.log(N))
        else:
            k = 1.86  # assumes N = 3*3600 / 10.8
        hmax = k * self.hs()
        hmax.attrs.update(
            {
                "standard_name": self._standard_name(self._my_name()),
                "units": self._units(self._my_name()),
            }
        )
        return hmax.rename(self._my_name())

    def scale_by_hs(
        self,
        expr,
        hs_min=-np.inf,
        hs_max=np.inf,
        tp_min=-np.inf,
        tp_max=np.inf,
        dpm_min=-np.inf,
        dpm_max=np.inf,
    ):
        """Scale spectra using expression based on Significant Wave Height hs.

        Args:
            - expr (str): expression to apply, e.g. '0.13*hs + 0.02'.
            - hs_min, hs_max, tp_min, tp_max, dpm_min, dpm_max (float): Ranges of hs,
                tp and dpm over which the scaling defined by `expr` is applied.

        """
        # Scale spectra by hs expression
        hs = self.hs()
        k = (eval(expr.lower()) / hs) ** 2
        scaled = k * self._obj

        # Condition over which scaling applies
        condition = True
        if hs_min != -np.inf or hs_max != np.inf:
            condition *= (hs >= hs_min) & (hs <= hs_max)
        if tp_min != -np.inf or tp_max != np.inf:
            tp = self.tp()
            condition *= (tp >= tp_min) & (tp <= tp_max)
        if dpm_min != -np.inf or dpm_max != np.inf:
            dpm = self.dpm()
            condition *= (dpm >= dpm_min) & (dpm <= dpm_max)

        return scaled.where(condition, self._obj)

    def tp(self, smooth=True):
        """Peak wave period Tp.

        Args:
            - smooth (bool): True for the smooth wave period, False for the discrete
              period corresponding to the maxima in the frequency spectra.

        """
        return xrstats.peak_wave_period(self._obj, smooth=smooth)

    def momf(self, mom=0):
        """Calculate given frequency moment."""
        fp = self.freq ** mom
        mf = self.dfarr * fp * self._obj
        return self._twod(mf.sum(dim=attrs.FREQNAME, skipna=False)).rename(
            f"mom{mom:0.0f}"
        )

    def momd(self, mom=0, theta=90.0):
        """Calculate given directional moment."""
        if self.dir is None:
            raise ValueError("Cannot calculate momd from 1d, frequency spectra.")
        cp = np.cos(np.radians(180 + theta - self.dir)) ** mom
        sp = np.sin(np.radians(180 + theta - self.dir)) ** mom
        msin = (self.dd * self._obj * sp).sum(dim=attrs.DIRNAME, skipna=False)
        mcos = (self.dd * self._obj * cp).sum(dim=attrs.DIRNAME, skipna=False)
        return msin, mcos

    def tm01(self):
        """Mean absolute wave period Tm01.

        True average period from the 1st spectral moment.

        """
        tm01 = self.momf(0).sum(dim=attrs.DIRNAME) / self.momf(1).sum(dim=attrs.DIRNAME)
        tm01.attrs.update(
            {
                "standard_name": self._standard_name(self._my_name()),
                "units": self._units(self._my_name()),
            }
        )
        return tm01.rename(self._my_name())

    def tm02(self):
        """Mean absolute wave period Tm02.

        Average period of zero up-crossings (Zhang, 2011).

        """
        tm02 = np.sqrt(
            self.momf(0).sum(dim=attrs.DIRNAME) / self.momf(2).sum(dim=attrs.DIRNAME)
        )
        tm02.attrs.update(
            {
                "standard_name": self._standard_name(self._my_name()),
                "units": self._units(self._my_name()),
            }
        )
        return tm02.rename(self._my_name())

    def dm(self):
        """Mean wave direction from the 1st spectral moment Dm."""
        if self.dir is None:
            raise ValueError("Cannot calculate dm from 1d, frequency spectra.")
        moms, momc = self.momd(1)
        dm = np.arctan2(
            moms.sum(dim=attrs.FREQNAME, skipna=False),
            momc.sum(dim=attrs.FREQNAME, skipna=False),
        )
        dm = (270 - R2D * dm) % 360.0
        dm.attrs.update(
            {
                "standard_name": self._standard_name(self._my_name()),
                "units": self._units(self._my_name()),
            }
        )
        return dm.rename(self._my_name())

    def dp(self):
        """Peak wave direction Dp.

        Defined as the direction where the energy density of the
        frequency-integrated spectrum is maximum.

        """
        return xrstats.peak_wave_direction(self._obj)

    def dpm(self):
        """Peak wave direction Dpm.

        Note From WW3 Manual:
            - peak wave direction, defined like the mean direction, using the
              frequency/wavenumber bin containing of the spectrum F(k) that contains
              the peak frequency only.

        """
        return xrstats.mean_direction_at_peak_wave_period(self._obj)

    def dspr(self):
        """Directional wave spreading Dspr.

        The one-sided directional width of the spectrum.

        """
        if self.dir is None:
            raise ValueError("Cannot calculate dspr from 1d, frequency spectra.")
        moms, momc = self.momd(1)
        # Manipulate dimensions so calculations work
        moms = self._twod(moms, dim=attrs.DIRNAME)
        momc = self._twod(momc, dim=attrs.DIRNAME)
        mom0 = self._twod(self.momf(0), dim=attrs.FREQNAME).spec.oned(skipna=False)

        dspr = (
            2
            * R2D ** 2
            * (1 - ((moms.spec.momf() ** 2 + momc.spec.momf() ** 2) ** 0.5 / mom0))
        ) ** 0.5
        dspr = dspr.sum(dim=attrs.FREQNAME, skipna=False).sum(
            dim=attrs.DIRNAME, skipna=False
        )
        dspr.attrs.update(
            {
                "standard_name": self._standard_name(self._my_name()),
                "units": self._units(self._my_name()),
            }
        )
        return dspr.rename(self._my_name())

    def crsd(self, theta=90.0):
        """Add description."""
        cp = np.cos(D2R * (180 + theta - self.dir))
        sp = np.sin(D2R * (180 + theta - self.dir))
        crsd = (self.dd * self._obj * cp * sp).sum(dim=attrs.DIRNAME)
        crsd.attrs.update(
            {
                "standard_name": self._standard_name(self._my_name()),
                "units": self._units(self._my_name()),
            }
        )
        return crsd.rename(self._my_name())

    def swe(self):
        """Spectral width parameter by Cartwright and Longuet-Higgins (1956).

        Represents the range of frequencies where the dominant energy exists.

        Reference:
            - Cartwright and Longuet-Higgins (1956). The statistical distribution
              of maxima of a random function. Proc. R. Soc. A237, 212-232.

        """
        swe = (
            1.0
            - self.momf(2).sum(dim=attrs.DIRNAME) ** 2
            / (
                self.momf(0).sum(dim=attrs.DIRNAME)
                * self.momf(4).sum(dim=attrs.DIRNAME)
            )
        ) ** 0.5
        swe = swe.where(swe >= 0.001, 1.0)
        swe.attrs.update(
            {
                "standard_name": self._standard_name(self._my_name()),
                "units": self._units(self._my_name()),
            }
        )
        return swe.rename(self._my_name())

    def sw(self, mask=np.nan):
        """Spectral width parameter by Longuet-Higgins (1975).

        Represents energy distribution over entire frequency range.

        Args:
            - mask (float): value for missing data in output

        Reference:
            - Longuet-Higgins (1975). On the joint distribution of the periods and
              amplitudes of sea waves. JGR, 80, 2688-2694.

        """
        sw = (
            self.momf(0).sum(dim=attrs.DIRNAME)
            * self.momf(2).sum(dim=attrs.DIRNAME)
            / self.momf(1).sum(dim=attrs.DIRNAME) ** 2
            - 1.0
        ) ** 0.5
        sw.attrs.update(
            {
                "standard_name": self._standard_name(self._my_name()),
                "units": self._units(self._my_name()),
            }
        )
        return sw.where(self.hs() >= 0.001).fillna(mask).rename(self._my_name())

    def celerity(self, depth=None):
        """Wave celerity C from frequency coords.

        Args:
            - depth (float): Water depth, use deep water approximation by default.

        Returns;
            - C: ndarray of same shape as freq with wave celerity for each frequency.

        """
        C = celerity(freq=self.freq, depth=depth)
        C.name = "celerity"
        return C

    def wavelen(self, depth=None):
        """Wavelength L from frequency coords.

        Args:
            - depth (float): Water depth, use deep water approximation by default.

        Returns;
            - L: ndarray of same shape as freq with wavelength for each frequency.

        """
        L = wavelen(freq=self.freq, depth=depth)
        L.name = "wavelength"
        return L

    def partition(
        self,
        wsp_darr,
        wdir_darr,
        dep_darr,
        swells=3,
        agefac=1.7,
        wscut=0.3333,
    ):
        """Partition wave spectra using Hanson's watershed algorithm.

        This method is not lazy, make sure array will fit into memory.

        Args:
            - wsp_darr (DataArray): wind speed (m/s).
            - wdir_darr (DataArray): Wind direction (degree).
            - dep_darr (DataArray): Water depth (m).
            - swells (int): Number of swell partitions to compute.
            - agefac (float): Age factor.
            - wscut (float): Wind speed cutoff.

        Returns:
            - part_spec (SpecArray): partitioned spectra with one extra dimension
              representig partition number.

        Note:
            - Input DataArrays must have same non-spectral dims as SpecArray.

        References:
            - Hanson, Jeffrey L., et al. "Pacific hindcast performance of three
              numerical wave models." JTECH 26.8 (2009): 1614-1633.

        """
        # Assert expected dimensions are defined
        if not {attrs.FREQNAME, attrs.DIRNAME}.issubset(self._obj.dims):
            raise ValueError(f"(freq, dir) dims required, only found {self._obj.dims}")
        for darr in (wsp_darr, wdir_darr, dep_darr):
            if set(darr.dims) != self._non_spec_dims:
                raise ValueError(
                    f"{darr.name} dims {list(darr.dims)} need matching "
                    f"non-spectral dims in SpecArray {self._non_spec_dims}"
                )

        return partition(
            dset=self._obj,
            wspd=wsp_darr,
            wdir=wdir_darr,
            dpt=dep_darr,
            swells=swells,
            agefac=agefac,
            wscut=wscut,
        )

    def stats(self, stats, fmin=None, fmax=None, dmin=None, dmax=None, names=None):
        """Calculate multiple spectral stats into a Dataset.

        Args:
            - stats (list): strings specifying stats to be calculated.
              (dict): keys are stats names, vals are dicts with kwargs to use with
              corresponding method.
            - fmin (float): lower frequencies for splitting spectra before
              calculating stats.
            - fmax (float): upper frequencies for splitting spectra before
              calculating stats.
            - dmin (float): lower directions for splitting spectra before
              calculating stats.
            - dmax (float): upper directions for splitting spectra before
              calculating stats.
            - names (list): strings to rename each stat in output Dataset.

        Returns:
            - Dataset with all spectral statistics specified.

        Note:
            - All stats names must correspond to methods implemented in this class.
            - If names is provided, its length must correspond to the length of stats.

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
            raise ValueError("stats must be either a container or a dictionary")

        names = names or stats_dict.keys()
        if len(names) != len(stats_dict):
            raise ValueError(
                "length of names does not correspond to the number of stats"
            )

        params = list()
        for func, kwargs in stats_dict.items():
            try:
                stats_func = getattr(spectra.spec, func)
            except AttributeError as err:
                raise ValueError(
                    "%s is not implemented as a method in %s"
                    % (func, self.__class__.__name__)
                ) from err
            if callable(stats_func):
                params.append(stats_func(**kwargs))
            else:
                raise ValueError(
                    "%s attribute of %s is not callable"
                    % (func, self.__class__.__name__)
                )

        return xr.merge(params).rename(dict(zip(stats_dict.keys(), names)))
