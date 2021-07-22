"""Wave spectra stats on numpy arrays sourced by apply_ufuncs."""
import numpy as np
from numba import guvectorize

from wavespectra.core.utils import D2R, R2D


def hs(spectrum, freq, dir, tail=True):
    """Significant wave height Hmo.

    Args:
        - spectrum (2darray): wave spectrum array.
        - freq (1darray): wave frequency array.
        - dir (1darray): wave direction array.
        - tail (bool): if True fit high-frequency tail before integrating spectra.

    """
    df = abs(freq[1:] - freq[:-1])
    if len(dir) > 1:
        ddir = abs(dir[1] - dir[0])
        E = ddir * spectrum.sum(1)
    else:
        E = np.squeeze(spectrum)
    Etot = 0.5 * sum(df * (E[1:] + E[:-1]))
    if tail and freq[-1] > 0.333:
        Etot += 0.25 * E[-1] * freq[-1]
    return 4.0 * np.sqrt(Etot)


@guvectorize(
    "(int64, float64[:], float64[:], float32[:])",
    "(), (n), (n) -> ()",
    nopython=True,
    target="parallel",
    cache=True,
    forceobj=True,
)
def dpm_gufunc(ipeak, momsin, momcos, out):
    """Mean direction at the peak wave period Dpm.

    Args:
        - ipeak (int): Index of the maximum energy density in the frequency spectrum E(f).
        - momsin (1darray): Sin component of the 1st directional moment.
        - momcos (1darray): Cos component of the 1st directional moment.

    Returns:
        - dpm (float): Mean direction at the frequency peak of the spectrum.

    """
    if not ipeak:
        out[0] = np.nan
    else:
        dpm = np.arctan2(momsin[ipeak], momcos[ipeak])
        out[0] = np.float32((270 - R2D * dpm) % 360.)


@guvectorize(
    "(int64, float32[:], float32[:])",
    "(), (n) -> ()",
    nopython=True,
    target="parallel",
    cache=True,
    forceobj=True,
)
def dp_gufunc(ipeak, dir, out):
    """Peak wave direction Dp.

    Args:
        - ipeak (int): Index of the maximum energy density in the frequency spectrum E(f).
        - dir (1darray): Wave direction array.

    Returns:
        - dp (float): Direction of the maximum energy density in the
          frequency-integrated spectrum.

    """
    out[0] = np.float32(dir[ipeak])


@guvectorize(
    "(int64, float64[:], float32[:], float32[:])",
    "(), (n), (n) -> ()",
    nopython=True,
    target="parallel",
    cache=True,
    forceobj=True,
)
def tps_gufunc(ipeak, spectrum, freq, out):
    """Smooth peak wave period Tp.

    Args:
        - ipeak (int): Index of the maximum energy density in frequency spectrum E(f).
        - spectrum (1darray): Direction-integrated wave spectrum array E(f).
        - freq (1darray): Wave frequency array.

    Returns:
        - tp (float): Period of the maximum energy density in the smooth spectrum.

    Note:
        - The smooth peak period is the peak of a parabolic fit around the spectral
          peak. It is the period commonly defined in SWAN and WW3 model output.

    """
    if not ipeak:
        out[0] = np.nan
    else:
        f1 = freq[ipeak - 1]
        f2 = freq[ipeak]
        f3 = freq[ipeak + 1]
        e1 = spectrum[ipeak - 1]
        e2 = spectrum[ipeak]
        e3 = spectrum[ipeak + 1]
        s12 = f1 + f2
        q12 = (e1 - e2) / (f1 - f2)
        q13 = (e1 - e3) / (f1 - f3)
        qa = (q13 - q12) / (f3 - f2)
        fp = (s12 - q12 / qa) / 2.0
        out[0] = np.float32(1.0 / fp)


@guvectorize(
    "(int64, float64[:], float32[:], float32[:])",
    "(), (n), (n) -> ()",
    nopython=True,
    target="parallel",
    cache=True,
    forceobj=True,
)
def tp_gufunc(ipeak, spectrum, freq, out):
    """Peak wave period Tp.

    Args:
        - ipeak (int): Index of the maximum energy density in frequency spectrum E(f).
        - spectrum (1darray): Frequency wave spectrum array E(f).
        - freq (1darray): Wave frequency array.

    Returns:
        - tp (float): Period of the maximum energy density in the frequency spectrum.

    Note:
        - Arg spectrum is only defined so the signature is consistent with tps function.

    """
    if not ipeak:
        out[0] = np.nan
    else:
        out[0] = np.float32(1.0 / freq[ipeak])
