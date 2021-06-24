import numpy as np
import xarray as xr
from scipy.constants import g, pi

from wavespectra import SpecArray
from wavespectra.core.utils import D2R, R2D, wavenuma, scaled, check_same_coordinates
from wavespectra.core.attributes import attrs, set_spec_attributes


def cartwright(dir, dm, dspr):
    """Cosine-squared directional spreading function of Cartwright (1963).

    Args:
        - dir (DataArray): Wave directions (degree).
        - dm: (DataArray, float): Mean wave direction (degree).
        - dspr (DataArray, float) Directional spreading (degree).

    Returns:
        - gth (DataArray): Spread function.

    Note:
        - If `dm` and `dspr` are DataArrays they must share the same coordinates.

    """
    check_same_coordinates(dm, dspr)

    # Angular difference from mean direction:
    dth = np.abs(dir - dm)
    dth = dth.where(dth <= 180, 360.0 - dth)  # for directional wrapping

    # spread function:
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
