"""Gaussian spectrum."""
import numpy as np
import xarray as xr
from scipy.constants import g, pi

import matplotlib.pyplot as plt

from wavespectra import SpecArray
from wavespectra.core.utils import scaled, check_same_coordinates
from wavespectra.core.attributes import attrs


def gaussian(freq, hs, fp, tm01, tm02):
    """Gaussian frequency spectrum (Bunney et al., 2014).

    Args:
        freq (DataArray): Frequency array (Hz).
        hs (DataArray, float): Significant wave height (m).
        fp (DataArray, float): Peak wave frequency (Hz).
        tm01 (DataArray, float): Mean wave period Tm.
        tm02 (DataArray, float): Zero-upcrossing wave period Tz.

    Returns:
        efth (SpecArray): Gaussian frequency spectrum E(f) (m2s).

    Note:
        If two or more input args other than `freq` are DataArrays,
            they must share the same coordinates.

    """
    mo = (hs / 4) ** 2
    sigma = np.sqrt( (mo / tm02**2) - (mo**2 / tm01**2) )
    term1 = mo / (sigma * np.sqrt(2 * pi))
    term2 = np.exp( -((freq - fp)**2 / (2 * (sigma)**2)) )
    dsout = term1 * term2

    dsout = scaled(dsout, hs)
    dsout.name = attrs.SPECNAME

    return dsout


def gaussian_stefan(freq, hs, fp, fpspr):
    """Gaussian frequency spectrum as modified from Stefan's code.

    Args:
        freq (DataArray): Frequency array (Hz).
        hs (DataArray, float): Significant wave height (m).
        fp (DataArray, float): Peak wave frequency (Hz).
        fpspr (DataArray, float): peak frequency spreading

    Returns:
        efth (SpecArray): Gaussian frequency spectrum E(f) (m2s).

    Note:
        If two or more input args other than `freq` are DataArrays,
            they must share the same coordinates.

    """
    freq_rel = (freq - fp) / fpspr
    dsout = xr.where(abs(freq_rel) < 10, np.exp(-0.5 * (freq_rel ** 2)), 0)

    dsout = scaled(dsout, hs)
    dsout.name = attrs.SPECNAME

    return dsout


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from wavespectra import read_wavespectra

    dset = read_wavespectra("/source/spec_recon_code/examples/weuro-spec-201201.nc")

    # Fit Gaussian spectrum from parameters calculated from testing spectrum (which is not an actual swell)

    ds = dset.isel(time=0, site=0, drop=True).load().sortby("dir")

    freq_orcaflex = ds.freq
    fp_orcaflex = ds.spec.fp()

    freq = ds.freq * (2 * np.pi)
    fp = ds.spec.fp() * (2 * np.pi)

    fpspr = ds.spec.swe()
    tm01 = ds.spec.tm01()
    tm02 = ds.spec.tm02()
    hs = ds.spec.hs()

    bunney = gaussian_bunney(freq=freq, hs=hs, fp=fp, tm01=tm01, tm02=tm02)
    orcaflex = gaussian_orcaflex(freq=freq_orcaflex, hs=hs, fp=fp_orcaflex, tm01=tm01, tm02=tm02)
    stefan = gaussian_stefan(freq=freq, hs=hs, fp=fp, fpspr=fpspr)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    bunney.plot(ax=ax1, color=color, label="Bunney", linewidth=10)
    stefan.plot(ax=ax1, label="Stefan", linewidth=10)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Bunney / Stefan', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    plt.legend()

    ax2 = ax1.twinx()

    color = 'tab:blue'
    orcaflex.plot(ax=ax2, color=color, label="Orcaflex", linewidth=4)
    ax2.set_ylabel('Orcaflex', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # plt.legend()

    fig.tight_layout()
    plt.show()
