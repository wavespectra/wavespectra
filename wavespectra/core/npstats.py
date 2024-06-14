"""Wave spectra stats on numpy arrays sourced by apply_ufuncs."""
import numpy as np
from scipy.constants import g, pi

from wavespectra.core.utils import R2D


def mom1(spectrum, dir, theta=90.0):
    """First directional moment.

    Args:
        - spectrum (2darray): wave spectrum array.
        - dir (1darray): wave direction array.
        - theta (float): angle offset.

    Returns:
        - msin (float): Sin component of the 1st directional moment.
        - mcos (float): Cosine component of the 1st directional moment.

    """
    dd = dir[1] - dir[0]
    cp = np.cos(np.radians(180 + theta - dir))
    sp = np.sin(np.radians(180 + theta - dir))
    msin = (dd * spectrum * sp).sum(axis=1)
    mcos = (dd * spectrum * cp).sum(axis=1)
    return msin, mcos


def dm(spectrum, dir):
    """Mean wave direction Dm.

    Args:
        - spectrum (2darray): wave spectrum array.
        - dir (1darray): wave direction array.

    Returns:
        - dm (float): Mean spectral period.

    """
    moms, momc = mom1(spectrum, dir)
    dm = np.arctan2(moms.sum(axis=0), momc.sum(axis=0))
    dm = (270 - R2D * dm) % 360.0
    return dm


def hs(spectrum, freq, dir=None, tail=True):
    """Significant wave height Hmo.

    Args:
        - spectrum (2darray): wave spectrum array.
        - freq (1darray): wave frequency array.
        - dir (1darray): wave direction array.
        - tail (bool): if True fit high-frequency tail before integrating spectra.

    Returns:
        - hs (float): Significant wave height.

    """
    df = abs(freq[1:] - freq[:-1])
    if dir is not None and len(dir) > 1:
        ddir = abs(dir[1] - dir[0])
        E = ddir * spectrum.sum(1)
    else:
        E = np.squeeze(spectrum)
    Etot = 0.5 * sum(df * (E[1:] + E[:-1]))
    if tail and freq[-1] > 0.333:
        Etot += 0.25 * E[-1] * freq[-1]
    return 4.0 * np.sqrt(Etot)


def dpm(ipeak, momsin, momcos):
    """Mean direction at the peak wave period Dpm.

    Args:
        - ipeak (int): Index of the maximum energy density in the frequency spectrum E(f).
        - momsin (1darray): Sin component of the 1st directional moment.
        - momcos (1darray): Cos component of the 1st directional moment.

    Returns:
        - dpm (float): Mean direction at the frequency peak of the spectrum.

    """
    if not ipeak:
        return np.nan
    else:
        dpm = np.arctan2(momsin[ipeak], momcos[ipeak])
        return np.float32((270 - R2D * dpm) % 360.0)


def dp(ipeak, dir):
    """Peak wave direction Dp.

    Args:
        - ipeak (int): Index of the maximum energy density in the direction spectrum E(d).
        - dir (1darray): Wave direction array.

    Returns:
        - dp (float): Direction of the maximum energy density in the
          frequency-integrated spectrum.

    """
    return np.float32(dir[ipeak])


def alpha(spectrum, freq, fp):
    """Phillips fetch dependant scaling coefficient.

    Args:
        - spectrum (1darray): Direction-integrated wave spectrum array E(f).
        - freq (1darray): Wave frequency array.
        - fp (float): Peak wave frequency (Hz).

    Returns:
        - alpha (float): Phillips constant.

    """
    # Positions for fitting high-frequency tail
    pos = np.where((freq > 1.35 * fp) & (freq < 2.0 * fp))[0]
    if pos.size == 0:
        pos = [freq.size - 2, freq.size - 1]
    elif pos.size == 1:
        if pos[0] == freq.size[-1]:
            pos = [pos[0] - 1, pos[0]]
        else:
            pos = [pos[0], pos[0] + 1]
    s = spectrum[pos]
    f = freq[pos]
    term1 = (2 * pi) ** 4 / g**2 / ((pos[-1] - pos[0]) + 1)
    term2 = np.sum(s * f**5 * np.exp(1.25 * (fp / f) ** 4))
    return np.float32(term1 * term2)


def tps(ipeak, spectrum, freq):
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
        return np.nan
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
        return np.float32(1.0 / fp)


def tp(ipeak, spectrum, freq):
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
        return np.nan
    else:
        return np.float32(1.0 / freq[ipeak])


def dpspr(ipeak, fdspr):
    """Peak directional wave spread Dpspr.

    Args:
        - ipeak (int): Index of the maximum energy density in the frequency spectrum E(f).
        - fdsprd (1darray): Direction spread as a function of frequency :math:`\\sigma(f)`.

    Returns:
        - dpspr (float): Directional wave spreading at the peak wave frequency.

    """
    if not ipeak:
        return np.nan
    else:
        return fdspr[ipeak]


def jonswap(freq, fpeak, hsig, gamma=3.3, alpha=0.0081, sigma_a=0.07, sigma_b=0.09):
    """Jonswap frequency spectrum for developing seas (Hasselmann et al., 1973).

    Args:
        - freq (1darray): Frequency array (Hz).
        - fpeak (float): Peak wave frequency (Hz).
        - hsig (float): Significant wave height (m), if provided the Jonswap
          spectra are scaled so that :math:`4\\sqrt{m_0} = hs`.
        - gamma (float): Peak enhancement parameter.
        - alpha (float): Phillip's fetch-dependent scaling coefficient.
        - sigma_a (float): width of the peak enhancement parameter for f <= fp.
        - sigma_b (float): width of the peak enhancement parameter for f > fp.

    Returns:
        - efth (SpecArray): Jonswap spectrum E(f) (m2s).

    Note:
        - This function is a numpy version of the `wavespectra.frequency.jonswap`
          function and is primarily defined for spectral fitting.
        - If `hs` is provided than the scaling parameter `alpha` becomes irrelevant.

    """
    sigma = np.where(freq <= fpeak, sigma_a, sigma_b)
    term1 = alpha * g**2 * (2 * pi) ** -4 * freq**-5
    term2 = np.exp(-(5 / 4) * (freq / fpeak) ** -4)
    term3 = gamma ** np.exp(-((freq - fpeak) ** 2) / (2 * sigma**2 * fpeak**2))
    dsout = term1 * term2 * term3
    if hsig is not None:
        dsout = dsout * (hsig / hs(dsout, freq)) ** 2
    return dsout


def gaussian(freq, fpeak, hsig, gw):
    """Gaussian frequency spectrum (Bunney et al., 2014).

    Args:
        - freq (1darray): Frequency array (Hz).
        - fpeak (float): Peak wave frequency (Hz).
        - hsig (float): Significant wave height (m).
        - gw (float): Gaussian width parameter :math:`\sigma` (m2s).

    Returns:
        - efth (SpecArray): Gaussian frequency spectrum E(f) (m2s).

    Note:
        - This function is a numpy version of the `wavespectra.frequency.gaussian`
          function and is primarily defined for spectral fitting.

    """
    mo = (hsig / 4) ** 2
    return mo / (gw * np.sqrt(2 * pi)) * np.exp(-0.5 * ((freq - fpeak) / gw) ** 2)
