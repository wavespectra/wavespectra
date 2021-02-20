"""Wave spectra stats on numpy arrays sourced by apply_ufuncs."""
import numpy as np


def dp(dirspec, dir):
    """Peak wave direction Dp.

    Args:
        - dirspec (1darray): Frequency-integrated wave spectrum array E(d).
        - freq (1darray): Wave frequency array.

    Returns:
        - dp (float): Direction of the maximum energy density in the
          frequency-integrated spectrum.

    """
    return dir[np.argmax(dirspec)]


def tps(spectrum, freq):
    """Smooth peak wave period Tp.

    Args:
        - spectrum (1darray): Direction-integrated wave spectrum array E(f).
        - freq (1darray): Wave frequency array.

    Returns:
        - tp (float): Period of the maximum energy density in the smooth spectrum.

    Note:
        - The smooth peak period is the peak of a parabolic fit around the spectral
          peak. It is the period commonly defined in SWAN and WW3 model output.

    """
    ipeak = fpeak(spectrum)
    if not ipeak:
        return None
    f1, f2, f3 = [freq[ipeak + i] for i in [-1, 0, 1]]
    e1, e2, e3 = [spectrum[ipeak + i] for i in [-1, 0, 1]]
    s12 = f1 + f2
    q12 = (e1 - e2) / (f1 - f2)
    q13 = (e1 - e3) / (f1 - f3)
    qa = (q13 - q12) / (f3 - f2)
    fp = (s12 - q12 / qa) / 2.0
    return 1.0 / fp


def tp(spectrum, freq):
    """Peak wave period Tp.

    Args:
        - spectrum (1darray): Frequency wave spectrum array E(f).
        - freq (1darray): Wave frequency array.

    Returns:
        - tp (float): Period of the maximum energy density in the frequency spectrum.

    """
    ipeak = fpeak(spectrum)
    if not ipeak:
        return None
    return 1.0 / freq[ipeak]


def fpeak(arr):
    """Returns the index of largest peak in the frequency spectrum.

    Args:
        - arr (1darray): Frequency spectrum.

    Returns:
        - ipeak (SpecArray): indices for slicing arr at the frequency peak.

    Note:
        - A peak is found when arr(ipeak-1) < arr(ipeak) < arr(ipeak+1).
        - Given the above, ipeak == 0 implies no peak has been detected.

    """
    ispeak = (np.diff(np.append(arr[0], arr)) > 0) & (
        np.diff(np.append(arr, arr[-1])) < 0
    )
    peak_pos = np.where(ispeak, arr, 0).argmax()
    return peak_pos


