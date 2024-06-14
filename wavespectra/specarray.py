"""Spectra object based on DataArray to calculate spectral statistics.

Reference:
    - Bunney, C., Saulter, A., Palmer, T. (2014). Reconstruction of complex 2D wave spectra for
      rapid deployment of nearshore wave models. From Sea to Shore—Meeting the Challenges of the
      Sea (Coasts, Marine Structures and Breakwaters 2013), W. Allsop and K. Burgess, Eds., 1050–1059.

    - Cartwright and Longuet-Higgins (1956). The Statistical Distribution of the Maxima
      of a Random Function, Proceedings of the Royal Society of London. Series A,
      Mathematical and Physical Sciences, 237, 212-232.

    - Hanson, Jeffrey L., et al. (2009). Pacific hindcast performance of three numerical wave models.
      JTECH 26.8 (2009): 1614-1633.

    - Holthuijsen LH (2005). Waves in oceanic and coastal waters (page 82).

    - Goda, Y. (1970). Numerical experiments on wave statistics with spectral simulation.
      Report of the Port and Harbour Research Institute 9, 3-57.

    - Longuet-Higgins (1975). On the joint distribution of the periods and amplitudes
      of sea waves, Journal of Geophysical Research, 80, 2688-2694.

    - Phillips (1957). On the generation of waves by turbulent wind, Journal of Fluid Mechanics, 2, pp 426-434.

"""
import re
import numpy as np
import xarray as xr
import inspect
import warnings

from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.utils import (
    D2R,
    R2D,
    celerity,
    wavenuma,
    wavelen,
    regrid_spec,
    smooth_spec,
)
from wavespectra.core import xrstats
from wavespectra.core.fitting import fit_jonswap_params, fit_gaussian_params
from wavespectra.plot import polar_plot
from wavespectra.partition.partition import Partition


@xr.register_dataarray_accessor("spec")
class SpecArray(object):
    """Extends DataArray with methods to deal with wave spectra."""

    def __init__(self, xarray_obj):
        """Initialise spec accessor."""
        self._obj = xarray_obj

        # These are set when property is first called to avoid computing more than once
        self._df = None
        self._dd = None

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
    def _spec_dims(self):
        """Return the set of spectral dimensions in underlying dataset."""
        return [d for d in self._obj.dims if d in [attrs.FREQNAME, attrs.DIRNAME]]

    @property
    def _non_spec_dims(self):
        """Return the set of non-spectral dimensions in underlying dataset."""
        return set(self._obj.dims).difference((attrs.FREQNAME, attrs.DIRNAME))

    @property
    def df(self):
        """Frequency resolution DataArray."""
        if self.freq.size > 1:
            return xr.DataArray(np.gradient(self.freq), coords=self.freq.coords)
        else:
            return xr.DataArray(data=np.array((1.0,)), coords=self.freq.coords)

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

    @property
    def partition(self):
        return Partition(self._obj)

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
        fwd = (
            xr.concat((arr.isel(freq=0), arr), dim=attrs.FREQNAME).diff(
                attrs.FREQNAME, n=1, label="upper"
            )
            > 0
        )
        bwd = (
            xr.concat((arr, arr.isel(freq=-1)), dim=attrs.FREQNAME).diff(
                attrs.FREQNAME, n=1, label="lower"
            )
            < 0
        )
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

    def _get_cf_attributes(self, name):
        """Returns dict with standard_name and units for method defined by name."""
        return {"standard_name": self._standard_name(name), "units": self._units(name)}

    def _is_circular(self):
        """Check if directions cover 360 degrees."""
        return (abs(self.dir.max() - self.dir.min() + self.dd - 360)) < (0.1 * self.dd)

    def oned(self, skipna=True):
        """Returns the one-dimensional frequency spectra.

        Direction dimension is dropped after integrating.

        Args:
            - skipna (bool): choose it to skip nans when integrating spectra.
              This is the default behaviour for sum() in DataArray. Notice it
              converts masks, where the entire array is nan, into zero.

        """
        if self.dir is not None:
            dsout = self.dd * self._obj.sum(dim=attrs.DIRNAME, skipna=skipna)
        else:
            dsout = self._obj.copy(deep=True)
        set_spec_attributes(dsout)
        return dsout

    def split(
        self, fmin=None, fmax=None, dmin=None, dmax=None, interpolate=True, rechunk=True
    ):
        """Split spectra over freq and/or dir dims.

        Args:
            - fmin (float): lowest frequency to split spectra, by default the lowest.
            - fmax (float): highest frequency to split spectra, by default the highest.
            - dmin (float): lowest direction to split spectra at, by default min(dir).
            - dmax (float): highest direction to split spectra at, by default max(dir).
            - interpolate (bool): Interpolate spectra at frequency cutoffs they
              are not available in coordinates of input spectra.
            - rechunk (bool): Rechunk split dims so there is one single chunk.

        """
        if fmax is not None and fmin is not None and fmax <= fmin:
            raise ValueError("fmax needs to be greater than fmin")
        if dmax is not None and dmin is not None and dmax <= dmin:
            raise ValueError("dmax needs to be greater than dmin")

        # Slice frequencies
        other = self._obj.sel(freq=slice(fmin, fmax))

        tol = 1e-10

        # Interpolate at fmin
        if interpolate and fmin is not None:
            if abs(float(other[attrs.FREQNAME][0]) - fmin) > tol:
                other = xr.concat([self._interp_freq(fmin), other], dim=attrs.FREQNAME)

        # Interpolate at fmax
        if interpolate and fmax is not None:
            if abs(float(other[attrs.FREQNAME][-1]) - fmax) > tol:
                other = xr.concat([other, self._interp_freq(fmax)], dim=attrs.FREQNAME)

        other.freq.attrs = self._obj[attrs.FREQNAME].attrs
        chunks = {attrs.FREQNAME: -1}

        # Slice directions
        if attrs.DIRNAME in other.dims and (dmin or dmax):
            other = other.sortby([attrs.DIRNAME]).sel(
                {attrs.DIRNAME: slice(dmin, dmax)}
            )
            other[attrs.DIRNAME].attrs = self._obj[attrs.DIRNAME].attrs
            chunks.update({attrs.DIRNAME: -1})

        if rechunk:
            other = other.chunk(chunks)

        return other

    def to_energy(self):
        """Convert spectra from energy density (m2/Hz/degree) into wave energy (m2)."""
        energy = self._obj * self.df * self.dd
        set_spec_attributes(energy)
        energy.attrs.update(self._get_cf_attributes(attrs.ESPECNAME))
        return energy.rename(attrs.ESPECNAME)

    def hs(self, tail=True):
        """Spectral significant wave height Hm0.

        Args:
            - tail (bool): if True fit high-frequency tail before integrating spectra.

        """
        Sf = self.oned(skipna=False)
        E = (Sf * self.df).sum(dim=attrs.FREQNAME)
        if tail and Sf.freq[-1] > 0.333:
            E += (
                0.25
                * Sf[{attrs.FREQNAME: -1}].drop_vars(attrs.FREQNAME)
                * Sf.freq[-1].values
            )
        hs = 4 * np.sqrt(E)
        hs.attrs.update(self._get_cf_attributes(self._my_name()))
        return hs.rename(self._my_name())

    def hrms(self, tail=True):
        """Root mean square wave height Hrms.

        Args:
            - tail (bool): if True fit high-frequency tail before integrating spectra.

        """
        Sf = self.oned(skipna=False)
        E = (Sf * self.df).sum(dim=attrs.FREQNAME)
        if tail and Sf.freq[-1] > 0.333:
            E += (
                0.25
                * Sf[{attrs.FREQNAME: -1}].drop_vars(attrs.FREQNAME)
                * Sf.freq[-1].values
            )
        hrms = np.sqrt(E * 8)
        hrms.attrs.update(self._get_cf_attributes(self._my_name()))
        return hrms.rename(self._my_name())

    def hmax(self):
        """Maximum wave height Hmax.

        hmax is the most probably value of the maximum individual wave height
        for each sea state. Note that maximum wave height can be higher (but
        not by much since the probability density function is rather narrow).

        Reference:
            - Holthuijsen (2005).

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
        hmax.attrs.update(self._get_cf_attributes(self._my_name()))
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
            condition *= ((hs >= hs_min) & (hs <= hs_max)).chunk()
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

        Note:
            - The smooth wave period is defined from a parabolic fit around the
              discrete peak of the frequency spectrum.

        """
        return xrstats.peak_wave_period(self.oned(), smooth=smooth)

    def fp(self, smooth=True):
        """Peak wave frequency Fp.

        Args:
            - smooth (bool): True for the smooth wave period, False for the discrete
              period corresponding to the maxima in the frequency spectrum.

        Note:
            - The smooth wave period is defined from a parabolic fit around the
              discrete peak of the frequency spectrum.

        """
        fp = 1 / self.tp(smooth=smooth)
        fp.attrs.update(self._get_cf_attributes(self._my_name()))
        return fp.rename(self._my_name())

    def momf(self, mom=0):
        """Frequency moment.

        Args:
            - mom (int): Moment to calculate.

        Returns:
            - mf (DataArray): The mth frequency moments for each direction.

        """
        fp = self.freq**mom
        mf = self.df * fp * self.oned()
        return mf.sum(dim=attrs.FREQNAME, skipna=False).rename(f"mom{mom:0.0f}")

    def momd(self, mom=0, theta=90.0):
        """Directional moment.

        Args:
            - mom (int): Moment to calculate.
            - theta (float): angle offset.

        Returns:
            - msin (DataArray): Sin component of the mth directional moment
              for each frequency.
            - mcos (DataArray): Cosine component of the mth directional moment
              for each frequency.

        """
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
        m0 = self.momf(0)
        m1 = self.momf(1)
        tm01 = m0 / m1
        tm01.attrs.update(self._get_cf_attributes(self._my_name()))
        return tm01.rename(self._my_name())

    def tm02(self):
        """Mean absolute wave period Tm02.

        Average period of zero up-crossings (Zhang, 2011).

        """
        m0 = self.momf(0)
        m2 = self.momf(2)
        tm02 = np.sqrt(m0 / m2)
        tm02.attrs.update(self._get_cf_attributes(self._my_name()))
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
        dm.attrs.update(self._get_cf_attributes(self._my_name()))
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
        """Mean directional wave spread Dspr.

        The one-sided directional width of the spectrum.

        """
        if self.dir is None:
            raise ValueError("Cannot calculate dspr from 1d, frequency spectra.")
        mom_sin, mom_cos = self.momd(1)
        a = (mom_sin * self.df).sum(dim=attrs.FREQNAME)
        b = (mom_cos * self.df).sum(dim=attrs.FREQNAME)
        e = (self.oned() * self.df).sum(dim=attrs.FREQNAME)
        dspr = (2 * R2D**2 * (1 - ((a**2 + b**2) ** 0.5 / e))) ** 0.5
        dspr.attrs.update(self._get_cf_attributes(self._my_name()))
        return dspr.rename(self._my_name())

    def fdspr(self, mom=1):
        """Directional wave spread at frequency :math:`dspr(f)`.

        The directional width of the spectrum at each frequency.

        Args:
            - mom (int): Directional moment for calculating the mth directional spread.

        """
        if self.dir is None:
            raise ValueError("Cannot calculate dpspr from 1d, frequency spectra.")
        mom_sin, mom_cos = self.momd(mom=mom)
        a = mom_sin * self.df
        b = mom_cos * self.df
        e = self.oned() * self.df
        fdspr = (2 * R2D**2 * (1 - ((a**2 + b**2) ** 0.5 / e))) ** 0.5
        return fdspr.rename(f"fdspr{mom:0.0f}")

    def dpspr(self, mom=1):
        """Peak directional wave spread Dpspr.

        The directional width of the spectrum at peak frequency.

        Args:
            - mom (int): Directional moment to calculate the mth directional spread.

        """
        return xrstats.peak_directional_spread(self._obj)

    def crsd(self, theta=90.0):
        """Add description."""
        cp = np.cos(D2R * (180 + theta - self.dir))
        sp = np.sin(D2R * (180 + theta - self.dir))
        crsd = (self.dd * self._obj * cp * sp).sum(dim=attrs.DIRNAME)
        crsd.attrs.update(self._get_cf_attributes(self._my_name()))
        return crsd.rename(self._my_name())

    def swe(self):
        """Spectral width parameter by Cartwright and Longuet-Higgins (1956).

        Represents the range of frequencies where the dominant energy exists.

        Reference:
            - Cartwright and Longuet-Higgins (1956).

        """
        m0 = self.momf(0)
        m2 = self.momf(2)
        m4 = self.momf(4)
        swe = (1.0 - m2**2 / (m0 * m4)) ** 0.5
        swe = swe.where(swe >= 0.001, 1.0)
        swe.attrs.update(self._get_cf_attributes(self._my_name()))
        return swe.rename(self._my_name())

    def sw(self):
        """Spectral width parameter by Longuet-Higgins (1975).

        Represents energy distribution over entire frequency range.

        Reference:
            - Longuet-Higgins (1975).

        """
        m0 = self.momf(0)
        m1 = self.momf(1)
        m2 = self.momf(2)
        sw = (m0 * m2 / m1**2 - 1.0) ** 0.5
        sw.attrs.update(self._get_cf_attributes(self._my_name()))
        return sw.where(self.hs() >= 0.001).rename(self._my_name())

    def gw(self):
        """Gaussian frequency spread by Bunney et al. (2014).

        Represents gaussian width of a swell partition.

        Reference:
            - Bunney et al. (2014).

        """
        m0 = (self.hs() / 4) ** 2
        gw = np.sqrt((m0 / (self.tm02() ** 2)) - (m0**2 / self.tm01() ** 2))
        gw.attrs.update(self._get_cf_attributes(self._my_name()))
        return gw.rename(self._my_name())

    def alpha(self, smooth=True):
        """Jonswap fetch dependant scaling coefficient alpha.

        Args:
            - smooth (bool): True for the smooth wave frequency, False for the discrete
              frequency corresponding to the peak in the frequency spectrum.

        Reference:
            - Phillips (1957).

        """
        return xrstats.alpha(self.oned(), smooth=smooth)

    def gamma(self, smooth=True, scaled=True):
        """Jonswap peak enhancement factor gamma.

        Represents the ratio between the peak in the frequency spectrum :math:`E(f)`
        and its associate Pierson-Moskowitz shape.

         Args:
            - smooth (bool): True for the smooth wave frequency, False for the discrete
              frequency corresponding to the peak in the frequency spectrum.
            - scaled (bool): True to apply polynomial approximation to gamma.

        """
        fp = self.fp(smooth=smooth)
        alpha_pm = 0.3125 * self.hs() ** 2 * fp**4
        epm_fp = alpha_pm * fp**-5 * 0.2865048
        gamma = self.oned().max(dim=attrs.FREQNAME) / epm_fp
        if scaled:
            # polynomial approximation for gamma
            p = [0.0378375, -0.13543292, 0.64087366, 0.32524949, 0.12974958]
            gamma_scaled = 0
            for pow, c in enumerate(p[::-1]):
                gamma_scaled += c * gamma**pow
            gamma = gamma_scaled
        gamma.attrs.update(self._get_cf_attributes(self._my_name()))
        return gamma.where(gamma >= 1, 1).rename(self._my_name())

    def goda(self):
        """Goda peakedness parameter.

        Reference:
            - Goda (1970).

        """
        ef = self.oned()
        mo2 = (ef * self.df).sum(dim=attrs.FREQNAME) ** 2
        goda = (2 / mo2) * (ef**2 * self.freq * self.df).sum(attrs.FREQNAME)
        goda.attrs.update(self._get_cf_attributes(self._my_name()))
        return goda.rename(self._my_name())

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

    def uss_x(self, depth=None, theta=90.0):
        """Stokes drift - x component, at sea surface. No high frequency tail.

        Args:
            - depth (float): Water depth, use deep water approximation by default.
            - theta (float): angle offset.

        """
        if self.dir is None:
            raise ValueError("Cannot calculate uss_x from 1d, frequency spectra.")
        if depth is None:
            L = 1.56 * (1.0 / self.freq) ** 2.0
            k = 2.0 * np.pi / L
        else:
            k = wavenuma(self.freq, depth)
        fk = 4.0 * np.pi * self.freq * k
        cp = np.cos(D2R * (180 + theta - self.dir))
        uss_x = (self.dd * fk * cp * self._obj * self.df).sum(
            dim=[attrs.FREQNAME, attrs.DIRNAME]
        )

        uss_x.attrs.update(self._get_cf_attributes(self._my_name()))
        return uss_x.rename(self._my_name())

    def uss_y(self, depth=None, theta=90.0):
        """Stokes drift - y component, at sea surface. No high frequency tail.

        Args:
            - depth (float): Water depth, use deep water approximation by default.
            - theta (float): angle offset.

        """

        if self.dir is None:
            raise ValueError("Cannot calculate uss_x from 1d, frequency spectra.")
        if depth is None:
            L = 1.56 * (1.0 / self.freq) ** 2.0
            k = 2.0 * np.pi / L
        else:
            k = wavenuma(self.freq, depth)
        fk = 4.0 * np.pi * self.freq * k
        sp = np.sin(D2R * (180 + theta - self.dir))
        uss_y = (self.dd * fk * sp * self._obj * self.df).sum(
            dim=[attrs.FREQNAME, attrs.DIRNAME]
        )

        uss_y.attrs.update(self._get_cf_attributes(self._my_name()))
        return uss_y.rename(self._my_name())

    def uss(self, depth=None):
        """Stokes drift - speed, at sea surface. No high frequency tail.

        Args:
            - depth (float): Water depth, use deep water approximation by default.

        """

        if depth is None:
            L = 1.56 * (1.0 / self.freq) ** 2.0
            k = 2.0 * np.pi / L
        else:
            k = wavenuma(self.freq, depth)
        fk = 4.0 * np.pi * self.freq * k
        uss = (self.dd * fk * self._obj * self.df).sum(
            dim=[attrs.FREQNAME, attrs.DIRNAME]
        )

        uss.attrs.update(self._get_cf_attributes(self._my_name()))
        return uss.rename(self._my_name())

    def mss(self, depth=None):
        """Mean squared slope of sea surface.

        Args:
            - depth (float): Water depth, use deep water approximation by default.

        """
        if depth is None:
            L = 1.56 * (1.0 / self.freq) ** 2.0
            k = 2.0 * np.pi / L
        else:
            k = wavenuma(self.freq, depth)

        Sf = self.oned(skipna=False)
        mss = (k**2.0 * Sf * self.df).sum(dim=attrs.FREQNAME)

        mss.attrs.update(self._get_cf_attributes(self._my_name()))
        return mss.rename(self._my_name())

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

    def rotate(self, angle) -> xr.DataArray:
        """Rotate spectra.

        Args:
            - angle (float): Angle in degrees to rotate spectra, positive for clockwise
              and negative for counter-clockwise rotation.

        Returns:
            - efth (DataArray): Rotated spectra.

        Note:
            - Directions are interpolated so that the rotated spectra have the same
              directional bins as the original spectra.

        """
        dirs = self.dir.values + angle
        dsout = self._obj.assign_coords({attrs.DIRNAME: dirs % 360})
        return regrid_spec(dsout, dir=self.dir)

    def smooth(self, freq_window=3, dir_window=3):
        """Smooth spectra with a running average.

        Args:
            - freq_window (int): Rolling window size along `freq` dim.
            - dir_window (int): Rolling window size along `dir` dim.

        Returns:
            - efth (DataArray): Smoothed spectra.

        Note:
            - Window sizes must be odd to ensure symmetry.

        """
        return smooth_spec(self._obj, freq_window=freq_window, dir_window=dir_window)

    def interp(self, freq=None, dir=None, maintain_m0=True):
        """Interpolate onto new spectral basis.

        Args:
            - freq (DataArray, 1darray): Frequencies of interpolated spectra (Hz).
            - dir (DataArray, 1darray): Directions of interpolated spectra (deg).
            - maintain_m0 (bool): Ensure variance is conserved in interpolated spectra.

        Returns:
            - dsi (DataArray): Regridded spectra.

        Note:
            - All freq below lowest freq are interpolated assuming :math:`E_d(f=0)=0`.
            - :math:`Ed(f)` is set to zero for new freq above the highest freq in dset.
            - Only the 'linear' method is currently supported.

        """
        return regrid_spec(self._obj, freq, dir, maintain_m0=maintain_m0)

    def interp_like(self, other, maintain_m0=True):
        """Interpolate onto coordinates from other spectra.

        Args:
            - other (Dataset, DataArray): Spectra defining new spectral basis.
            - maintain_m0 (bool): Ensure variance is conserved in interpolated spectra.

        Returns:
            - dsi (DataArray): Regridded spectra.

        """
        freq = getattr(other.spec, attrs.FREQNAME)
        dir = getattr(other.spec, attrs.DIRNAME)
        return self.interp(freq=freq, dir=dir, maintain_m0=maintain_m0)

    def plot(
        self,
        kind="contourf",
        normalised=True,
        logradius=True,
        as_period=False,
        rmin=None,
        rmax=None,
        show_theta_labels=True,
        show_radii_labels=True,
        radii_ticks=None,
        radii_labels_angle=22.5,
        radii_labels_size=8,
        cbar_ticks=None,
        cmap="RdBu_r",
        extend="neither",
        efth_min=1e-3,
        **kwargs,
    ):
        """Plot spectra in polar axis.

        Args:
            - kind (str): Plot kind, one of (`contourf`, `contour`, `pcolormesh`).
            - normalised (bool): Show efth normalised between 0 and 1.
            - logradius (bool): Set log radii.
            - as_period (bool): Set radii as wave period instead of frequency.
            - rmin (float): Minimum value to clip the radius axis.
            - rmax (float): Maximum value to clip the radius axis.
            - show_theta_labels (bool): Show direction tick labels.
            - show_radii_labels (bool): Show radii tick labels.
            - radii_ticks (array): Tick values for radii.
            - radii_labels_angle (float): Polar angle at which radii labels are positioned.
            - radii_labels_size (float): Fontsize for radii labels.
            - cbar_ticks (array): Tick values for colorbar.
            - cmap (str, obj): Colormap to use.
            - efth_min (float): Clip energy density below this value.
            - kwargs: All extra kwargs are passed to the plotting method defined by `kind`.

        Returns:
            - pobj: The xarray object returned by calling `da.plot.{kind}(**kwargs)`.

        Note:
            - If normalised==True, contourf uses a logarithmic colour scale by default.
            - Plot and axes can be redefined from the returned xarray object.
            - Xarray uses the `sharex`, `sharey` args to control which panels receive axis
              labels. In order to set labels for all panels, set these to `False`.
            - Masking of low values can be done in contourf by setting `efth_min` larger
              than the lowest contour level along with `extend` set to "neither" or "min".

        """
        return polar_plot(
            darr=self._obj.copy(deep=True),
            kind=kind,
            normalised=normalised,
            logradius=logradius,
            as_period=as_period,
            rmin=rmin,
            rmax=rmax,
            show_theta_labels=show_theta_labels,
            show_radii_labels=show_radii_labels,
            radii_ticks=radii_ticks,
            radii_labels_angle=radii_labels_angle,
            radii_labels_size=radii_labels_size,
            cbar_ticks=cbar_ticks,
            cmap=cmap,
            extend=extend,
            efth_min=efth_min,
            **kwargs,
        )

    def rmse(self, other):
        """Root-Mean-Square Error among spectral bins.

        Args:
            - other (SpecArray): Spectra to calculate binwise RMSE against.

        Note:
            - Spectral coordinates in self and other must be the same.
            - Non-spectral coordinates in self and other should ideally be the same.
              Coordinates are attempted to be broadcast if they are different.

        Reference:
            - Bunney et al. (2014).

        """
        e0 = self.to_energy()
        e1 = other.spec.to_energy()
        ediff = np.sqrt(((e0 - e1) ** 2).sum(dim=self._spec_dims))
        e0sum = np.sqrt(e0.sum(dim=self._spec_dims) ** 2)
        rmse = ediff / e0sum
        return rmse

    def fit_jonswap(self, gamma0=1.5, spectra=True, params=True):
        """Nonlinear fit Jonswap spectra.

        Args:
            - gamma0 (float): Initial guess for gamma.
            - spectra (bool): Return fitted spectra.
            - params (bool): Return fitted parameters.

        Return:
            - dsout (Dataset): Fitted Jonswap spectra and/or Jonswap parameters peak
              frequency, significant wave height and peak enhancement factor gamma.

        Note:
            - At least one of spectra or params must be True.

        """
        from wavespectra.construct.frequency import jonswap

        if not spectra and not params:
            raise ValueError("At least one of `spectra` or `params` must be True")

        fp, hs, gamma = xr.apply_ufunc(
            fit_jonswap_params,
            self.oned(),
            self.freq,
            self.fp(smooth=True),
            self.hs(),
            gamma0,
            input_core_dims=[[attrs.FREQNAME], [attrs.FREQNAME], [], [], []],
            output_core_dims=[[], [], []],
            exclude_dims=set((attrs.FREQNAME,)),
            dask="parallelized",
            vectorize=True,
            output_dtypes=["float32", "float32", "float32"],
        )
        dsout = xr.Dataset()
        if spectra:
            dsout["efth"] = jonswap(freq=self.freq, fp=fp, gamma=gamma, hs=hs)
        if params:
            dsout["fp"] = fp
            dsout["hs"] = hs
            dsout["gamma"] = gamma
        set_spec_attributes(dsout)
        dsout.attrs["title"] = "Fit Jonswap spectra"
        return dsout

    def fit_gaussian(self, gw0=1.5, spectra=True, params=True):
        """Nonlinear fit Gaussian width.

        Args:
            - gw0 (float): Initial guess for gw.
            - spectra (bool): Return fitted spectra.
            - params (bool): Return fitted parameters.

        Return:
            - dsout (Dataset): Fitted Gaussian spectra and/or Gaussian parameters peak
              frequency, significant wave height and gaussian width.

        Note:
            - At least one of spectra or params must be True.

        """
        from wavespectra.construct.frequency import gaussian

        if not spectra and not params:
            raise ValueError("At least one of `spectra` or `params` must be True")

        fp, hs, gw = xr.apply_ufunc(
            fit_gaussian_params,
            self.oned(),
            self.freq,
            self.fp(smooth=True),
            self.hs(),
            gw0,
            input_core_dims=[[attrs.FREQNAME], [attrs.FREQNAME], [], [], []],
            output_core_dims=[[], [], []],
            exclude_dims=set((attrs.FREQNAME,)),
            dask="parallelized",
            vectorize=True,
            output_dtypes=["float32", "float32", "float32"],
        )
        dsout = xr.Dataset()
        if spectra:
            dsout["efth"] = gaussian(freq=self.freq, hs=hs, fp=fp, gw=gw)
        if params:
            dsout["fp"] = fp
            dsout["hs"] = hs
            dsout["gw"] = gw
        set_spec_attributes(dsout)
        dsout.attrs["title"] = "Fit Gaussian spectra"
        return dsout
