import numpy as np
from scipy.optimize import curve_fit

from wavespectra.fit.jonswap import np_jonswap


def _fit_jonswap(ef, freq, fp0, hs0, gamma0=1.5):
    """Nonlinear fit Jonswap spectrum.

    Args:
        - ef (1darray): Frequency wave spectrum to fit (m2/Hz).
        - freq (1darray): Frequency array (Hz).
        - fp0 (float): Peak frequency first guess (Hz).
        - hs0 (float): Significant wave height first guess (m).
        - gamma0 (float): Peak enhancement factor first guess.

    Returns:
        - p1 (list): Fitted values for hs, fp and gamma.

    """
    p1, cov = curve_fit(
        f=np_jonswap,
        xdata=freq,
        ydata=ef,
        p0=[fp0, hs0, gamma0],
    )
    return p1


def fit_jonswap_spectra(ef, freq, fp0, hs0, gamma0):
    """Wrapper to return only spectrum from _fit_jonswap to run as ufunc."""
    fp, hs, gamma = _fit_jonswap(ef, freq, fp0, hs0, gamma0)
    return np_jonswap(freq, fp, hs, gamma)


def fit_jonswap_gamma(ef, freq, fp0, hs0, gamma0):
    """Wrapper to return only gamma from _fit_jonswap to run as ufunc."""
    return _fit_jonswap(ef, freq, fp0, hs0, gamma0)[-1]
