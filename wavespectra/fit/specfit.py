import numpy as np
import xarray as xr
from scipy.constants import g, pi

from wavespectra import SpecArray
from wavespectra.core.utils import D2R, R2D, wavenuma, scaled, check_same_coordinates
from wavespectra.core.attributes import attrs, set_spec_attributes


def pierson_moskowitz(hs, tp, freq):
    """Pierson and Moskowitz frequency spectrum (Pierson and Moskowitz, 1964)).

    Args:
        hs (DataArray, float): Significant wave height (m).
        tp (DataArray, float): Peak wave period (s).
        freq (DataArray, 1darray): Frequency array (Hz).

    Returns:
        efth (SpecArray): One-dimension Pierson-Moskowitz spectrum E(f) (m2s).

    Note:
        If hs and tp are DataArrays they must share the same coordinates.

    """
    check_same_coordinates(hs, tp)

    b = (tp / 1.057) ** -4
    a = b * (hs / 2) ** 2
    dsout = a * freq ** -5 * np.exp(-b * freq ** -4)

    dsout = scaled(dsout, hs)
    dsout.name = attrs.SPECNAME

    return dsout


def jonswap(hs, tp, freq, alpha=0.0081, gamma=3.3, sigma_a=0.07, sigma_b=0.09):
    """Jonswap frequency spectrum (Hasselmann et al., 1973).

    Args:
        hs (DataArray, float): Significant wave height (m).
        tp (DataArray, float): Peak wave period (s).
        freq (DataArray, 1darray): Frequency array (Hz).
        alpha (DataArray, float): Phillip's fetch-dependent scaling coefficient.
        gamma (DataArray, float): Peak enhancement parameter.
        sigma_a (float): width of the peak enhancement parameter for f <= fp.
        sigma_b (float): width of the peak enhancement parameter for f > fp.

    Returns:
        efth (SpecArray): One-dimension Pierson-Moskowitz spectrum E(f) (m2s).

    Note:
        If hs and tp are DataArrays they must share the same coordinates.

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


def tma(hs, tp, dep, freq, alpha=0.0081, gamma=3.3, sigma_a=0.07, sigma_b=0.09):
    """TMA frequency spectrum (Bouws et al., 1985).

    Args:
        hs (DataArray, float): Significant wave height (m).
        tp (DataArray, float): Peak wave period (s).
        dep (DataArray, float): Water depth (m).
        freq (DataArray, 1darray): Frequency array (Hz).
        alpha (DataArray, float): Phillip's fetch-dependent scaling coefficient.
        gamma (DataArray, float): Peak enhancement parameter.
        sigma_a (float): width of the peak enhancement parameter for f <= fp.
        sigma_b (float): width of the peak enhancement parameter for f > fp.

    Returns:
        efth (SpecArray): One-dimension Pierson-Moskowitz spectrum E(f) (m2s).

    Note:
        If hs, tp, dep are DataArrays they must share the same coordinates.

    """
    check_same_coordinates(hs, tp, dep)

    dsout = jonswap(hs, tp, freq, alpha, gamma, sigma_a, sigma_b)
    k = wavenuma(freq, dep)
    phi = np.tanh(k * dep) ** 2 / (1 + (2 * k * dep) / np.sinh(2 * k * dep))
    dsout = dsout * phi

    dsout = scaled(dsout, hs)
    dsout.name = attrs.SPECNAME

    return dsout


def gaussian(hs, freq, fp, fpspr, tm01, tm02):
    """Gaussian frequency spectrum.

    Args:
        f: frequencies
        fp: peak frequency
        fpspr: peak frequency spreading

    Output:
        Efth: sea surface variance density spectrum (len(f), len(th))

    """
    # As described in Bunney's
    mo = (hs / 4)**2
    sigma = np.sqrt( (mo / tm02**2) - (mo**2 / tm01**2) )
    term1 = mo / (2 * pi * sigma * np.sqrt(2 * pi))
    term2 = np.exp( -((freq - fp)**2 / (2 * (2 * pi * sigma)**2)) )
    ef1 = term1 * term2

    # Modified from Stefan's
    freq_rel = (freq - fp) / fpspr
    ef2 = xr.where(abs(freq_rel) < 10, np.exp(-0.5 * (freq_rel ** 2)), 0)

    # Plotting two formulas to compare
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Bunney', color=color)
    ax1.plot(freq, ef1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Stefan', color=color)
    ax2.plot(freq, ef2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

    # Returning Stefan's for now
    ef = ef2

    return ef


def cartwright(dir, dm, dspr):
    """Directional spreading function (Cartwright, 1963).

    Args:
        dir (DataArray): Wave directions (degrees).
        dm: (DataArray, float): Mean wave direction (degrees).
        dspr (DataArray, float) Directional spreading (degrees).

    Returns:
        gth (DataArray): Spread function.

    Note:
        If dm and dspr are DataArrays they must share the same coordinates.

    """
    check_same_coordinates(dm, dspr)

    # Angular difference from mean direction:
    dth = np.abs(dir - dm)
    dth = dth.where(dth <= 180, 360.0 - dth)  # for directional wrapping

    # spreading function:
    s = 2.0 / (np.deg2rad(dspr) ** 2) - 1
    gth = np.cos(0.5 * np.deg2rad(dth)) ** (2 * s)
    gth = gth.where(np.abs(dth) <= 90.0, 0.0)  # mask directions +-90 deg from mean

    # normalise
    gsum = 1.0 / (gth.sum(attrs.DIRNAME) * (2 * pi / dir.size))
    gth = gth * gsum

    return gth / R2D


def skewed(dir, dpm, dm, fp, fm):
    """Skewed, frequency-dependent directional spreading (Bunney et al., 2014)."""
    # dtf = (dpm - dm) / (fp - fm)
    pass


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import cmocean
    from wavespectra import read_wavespectra

    dset = read_wavespectra("/source/spec_recon_code/examples/weuro-spec-201201.nc")

    ds = dset.isel(time=0, site=0, drop=True).load().sortby("dir")
    ds.attrs = {}

    dspr = 30
    dm = 20
    hs = 2
    tp = 10
    dep = 15
    freq = ds.freq

    # 1D spectra
    sa1 = pierson_moskowitz(hs, tp, freq)
    sa2 = jonswap(hs, tp, freq)
    sa3 = jonswap(hs, tp, freq, gamma=1.0)
    sa4 = tma(hs, tp, dep, freq)
    sa5 = tma(hs, tp, 300, freq)

    # 2D spectra
    gth = cartwright(ds.dir, dm, dspr)
    Sa1 = sa1 * gth
    Sa2 = sa2 * gth
    Sa3 = sa3 * gth

    # Plotting 1d spectra
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sa1.plot(color="k", label="Pierson and Moskowitz", linewidth=3)
    sa2.plot(color="b", label="Jonswap (gamma=3.3)", linewidth=3)
    sa3.plot(color="r", label="Jonswap (gamma=1.0)", linewidth=1)
    sa4.plot(color="m", label="TMA (depth=15)", linewidth=2)
    sa5.plot(color="c", label="TMA (depth=300)", linewidth=1)
    plt.legend()

    # Plotting 2d spectra
    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot(131, projection="polar")
    Sa1.spec.plot.contourf(ax=ax, as_log10=True, vmin=-5, vmax=0, cmap="cividis")
    ax.set_title("Pierson and Moskowitz")
    ax = fig.add_subplot(132, projection="polar")
    Sa2.spec.plot.contourf(ax=ax, as_log10=True, vmin=-5, vmax=0, cmap="cividis")
    ax.set_title("Jonswap (gamma=3.3)")
    ax = fig.add_subplot(133, projection="polar")
    Sa3.spec.plot.contourf(ax=ax, as_log10=True, vmin=-5, vmax=0, cmap="cividis")
    ax.set_title("Jonswap (gamma=1.0)")

    plt.show()
