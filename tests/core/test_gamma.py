"""Testing the Jonswap peak enhancement factor gamma."""

import os
import pytest
import numpy as np
import xarray as xr

from wavespectra import read_swan
from wavespectra.core import npstats, xrstats
from wavespectra.construct.frequency import jonswap


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def freq():
    """Frequencies with the 10% increments typical of spectral wave models."""
    values = 0.03453 * 1.1 ** np.arange(30)
    return xr.DataArray(values, coords={"freq": values}, dims="freq")


@pytest.fixture(scope="module")
def fp(freq):
    """Peak frequencies spanning the range between two frequency bins."""
    return xr.DataArray(np.linspace(float(freq[9]), float(freq[10]), 21), dims="pos")


@pytest.fixture(scope="module")
def dset():
    return read_swan(os.path.join(FILES_DIR, "swanfile.spec"))


def parabola(freq, fp, ep, curvature=-2000.0):
    """Spectrum that is a parabola around the peak, zero elsewhere."""
    return (ep + curvature * (freq - fp) ** 2).clip(0)


@pytest.mark.parametrize("smooth", [True, False])
def test_output(dset, smooth):
    gamma = dset.spec.gamma(smooth=smooth)
    assert gamma.name == "gamma"
    assert set(gamma.dims) == set(dset.efth.dims) - {"freq", "dir"}
    assert (gamma >= 1).all()


@pytest.mark.parametrize("smooth", [True, False])
def test_peak_at_frequency_bin(freq, smooth):
    ef = jonswap(freq=freq, fp=float(freq[9]), hs=2.0, gamma=3.3)
    assert float(ef.spec.gamma(smooth=smooth)) == pytest.approx(3.3, rel=0.02)


@pytest.mark.parametrize("gamma", [2.0, 3.3, 7.0])
def test_smooth_is_less_sensitive_to_peak_position(freq, fp, gamma):
    """The largest bin reads the peak low if it is between bins, see issue #179."""
    ef = jonswap(freq=freq, fp=fp, hs=2.0, gamma=gamma)
    smooth = abs(ef.spec.gamma(smooth=True) / gamma - 1).max()
    raw = abs(ef.spec.gamma(smooth=False) / gamma - 1).max()
    assert smooth < 0.9 * raw


@pytest.mark.parametrize("gamma", [2.0, 3.3, 7.0])
def test_log_parabola_is_less_sensitive_to_peak_position(freq, fp, gamma):
    ef = jonswap(freq=freq, fp=fp, hs=2.0, gamma=gamma)
    log = abs(ef.spec.gamma(log_parabola=True) / gamma - 1).max()
    linear = abs(ef.spec.gamma(log_parabola=False) / gamma - 1).max()
    assert log < 0.8 * linear


def test_log_parabola_requires_smooth(dset):
    with pytest.raises(ValueError, match="only supported with smooth=True"):
        dset.spec.gamma(smooth=False, log_parabola=True)


def test_log_parabola_peak(freq):
    """Log peak energy and period are the vertex of the parabola in log space."""
    lnf = np.log(freq)
    ef = 3.0 * np.exp(-((lnf - np.log(0.0835)) ** 2) / (2 * 0.2**2))
    assert float(xrstats.peak_energy(ef, log=True)) == pytest.approx(3.0)
    assert float(1 / xrstats.peak_wave_period(ef, log=True)) == pytest.approx(
        0.0835, rel=1e-5
    )
    assert float(xrstats.peak_energy(ef, log=False)) != pytest.approx(3.0, rel=1e-4)


def test_log_parabola_falls_back_if_no_energy_around_peak(freq):
    ef = np.zeros(freq.size)
    ef[9:11] = [2.0, 1.0]
    assert npstats.eps(9, ef, freq.values, log=True) == npstats.eps(9, ef, freq.values)
    assert npstats.tps(9, ef, freq.values, log=True) == npstats.tps(9, ef, freq.values)


def test_peak_energy_is_consistent_with_peak_period(freq):
    """Smooth peak energy and period are the vertex of the same parabola."""
    ef = parabola(freq, fp=0.0835, ep=3.0)
    assert float(xrstats.peak_energy(ef, smooth=True)) == pytest.approx(3.0)
    assert float(1 / xrstats.peak_wave_period(ef, smooth=True)) == pytest.approx(
        0.0835, rel=1e-5
    )


def test_peak_energy_raw_is_the_largest_bin(freq):
    ef = parabola(freq, fp=0.0835, ep=3.0)
    assert float(xrstats.peak_energy(ef, smooth=False)) == float(ef.max())
    assert float(xrstats.peak_energy(ef, smooth=False)) < 3.0


def test_peak_energy_from_dataset(dset):
    ef = dset.efth.spec.oned()
    xr.testing.assert_identical(
        xrstats.peak_energy(ef.to_dataset()), xrstats.peak_energy(ef)
    )


@pytest.mark.parametrize("func", [npstats.eps, npstats.ep])
def test_npstats_no_peak(freq, func):
    assert np.isnan(func(0, np.zeros(freq.size), freq.values))


@pytest.mark.parametrize("smooth", [True, False])
@pytest.mark.parametrize("fill", [0.0, np.nan])
def test_no_energy(freq, smooth, fill):
    ef = xr.full_like(jonswap(freq=freq, fp=0.1, hs=2.0), fill)
    assert float(ef.spec.gamma(smooth=smooth)) == 1.0


@pytest.mark.parametrize("kwargs", [{}, {"smooth": False}, {"log_parabola": True}])
def test_dask(dset, kwargs):
    gamma = dset.chunk({"time": 3}).spec.gamma(**kwargs)
    assert gamma.chunks is not None
    np.testing.assert_allclose(gamma.values, dset.spec.gamma(**kwargs).values)
