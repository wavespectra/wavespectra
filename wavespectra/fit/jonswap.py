"""Jonswap spectrum."""
import numpy as np
import xarray as xr
from scipy.constants import g, pi

from wavespectra import SpecArray
from wavespectra.core.utils import scaled, check_same_coordinates
from wavespectra.core.attributes import attrs


def jonswap(hs, tp, freq, alpha=0.0081, gamma=3.3, sigma_a=0.07, sigma_b=0.09):
    """Jonswap frequency spectrum (Hasselmann et al., 1973).

    Args:
        hs (DataArray, float): Significant wave height (m).
        tp (DataArray, float): Peak wave period (s).
        freq (DataArray): Frequency array (Hz).
        alpha (DataArray, float): Phillip's fetch-dependent scaling coefficient.
        gamma (DataArray, float): Peak enhancement parameter.
        sigma_a (float): width of the peak enhancement parameter for f <= fp.
        sigma_b (float): width of the peak enhancement parameter for f > fp.

    Returns:
        efth (SpecArray): One-dimension Pierson-Moskowitz spectrum E(f) (m2s).

    Note:
        If two or more input args are DataArrays they must share the same coordinates.

    """
    check_same_coordinates(hs, tp)

    fp = 1 / tp
    sigma = xr.full_like(freq, sigma_a).where(freq <= fp, sigma_b)
    term1 = alpha * g ** 2 * (2 * pi) ** -4 * freq ** -5
    term2 = np.exp(-(5 / 4) * (freq / fp) ** -4)
    term3 = gamma ** np.exp(-((freq - fp) ** 2) / (2 * sigma ** 2 * fp ** 2))
    dsout = term1 * term2 * term3

    dsout = scaled(dsout, hs)
    dsout.name = attrs.SPECNAME

    return dsout



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from wavespectra import read_wavespectra, read_swan

    dset = read_wavespectra("/source/spec_recon_code/examples/weuro-spec-201201.nc")

    ds = dset.isel(time=0, site=0, drop=True).load().sortby("dir")
    ds.attrs = {}

    hs = 2
    tp = 10
    freq = ds.freq

    # 1D spectra
    sa1 = jonswap(hs, tp, freq)
    sa2 = jonswap(hs, tp, freq, gamma=1.0)

    # Plotting 1d spectra
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sa1.plot(color="b", label="Jonswap (gamma=3.3)", linewidth=3)
    sa2.plot(color="r", label="Jonswap (gamma=1.0)", linewidth=1)
    plt.legend()


    dset = read_swan("../../docs/_static/swanfile.spec")
    hs = dset.spec.hs()
    tp = dset.spec.tp()
    
    ds = jonswap(
        hs=dset.spec.hs(),
        tp=dset.spec.tp(),
        freq=dset.freq,
        gamma=1.5
    )
    ds

    plt.show()
