import os
import pytest
import numpy as np

from wavespectra import read_swan
from wavespectra.core.npstats import hs, dpm, dp, tps, tp, dpspr, steepness
from wavespectra.construct.frequency import jonswap, pierson_moskowitz
from wavespectra.core.utils import create_frequencies, to_coords


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def dset():
    """Load SpecDset but skip test if matplotlib is not installed."""
    pytest.importorskip("matplotlib")
    dset = read_swan(os.path.join(FILES_DIR, "swanfile.spec"), as_site=True)
    return dset


def test_hs(dset):
    ds = dset.isel(time=0, site=0, drop=True)
    hs(spectrum=ds.efth.values, freq=ds.freq.values, dir=ds.dir.values, tail=True)
    hs(spectrum=ds.efth.values, freq=ds.freq.values, dir=ds.dir.values, tail=False)
    ds = ds.isel(dir=[0])
    hs(spectrum=ds.efth.values, freq=ds.freq.values, dir=ds.dir.values)


def test_dpm(dset):
    ds = dset.isel(time=0, site=0, drop=True)
    ipeak = ds.efth.spec._peak(ds.efth.spec.oned())
    momsin, momcos = ds.spec.momd(1)
    out = dpm(int(ipeak), momsin.values, momcos.values)
    assert np.isclose(out, 249.09263611)
    out = dpm(0, momsin.values, momcos.values)
    assert np.isnan(out)


def test_dp(dset):
    ds = dset.isel(time=0, site=0, drop=True)
    ipeak = np.int64(ds.efth.spec._peak(ds.efth.spec.oned()))
    dir = ds.dir.values.astype("float32")
    out = dp(ipeak, dir)
    assert np.isclose(out, 55)


def test_tps(dset):
    ds = dset.isel(time=0, site=0, drop=True)
    ipeak = np.int64(ds.efth.spec._peak(ds.efth.spec.oned()))
    spectrum = ds.efth.spec.oned().values.astype("float64")
    freq = ds.freq.values.astype("float32")
    out = tps(ipeak, spectrum, freq)
    assert np.isclose(out, 12.907742)
    out = tps(0, spectrum, freq)
    assert np.isnan(out)


def test_tp(dset):
    ds = dset.isel(time=0, site=0, drop=True)
    ipeak = np.int64(ds.efth.spec._peak(ds.efth.spec.oned()))
    spectrum = ds.efth.spec.oned().values.astype("float64")
    freq = ds.freq.values.astype("float32")
    out = tp(ipeak, spectrum, freq)
    assert np.isclose(out, 13.568521)
    out = tp(0, spectrum, freq)
    assert np.isnan(out)


def test_dpspr(dset):
    ds = dset.isel(time=0, site=0, drop=True)
    ipeak = np.int64(ds.efth.spec._peak(ds.efth.spec.oned()))
    fdspr1 = ds.spec.fdspr(mom=1).values.astype("float64")
    fdspr2 = ds.spec.fdspr(mom=2).values.astype("float64")

    out = dpspr(ipeak, fdspr1)
    assert np.isclose(out, 8.463889)

    out = dpspr(ipeak, fdspr2)
    assert np.isclose(out, 29.384691)

    out = dpspr(0, fdspr1)
    assert np.isnan(out)


@pytest.fixture(scope="module")
def freq():
    """Well resolved frequency array."""
    return to_coords(create_frequencies(0.03, 50, 1.1), "freq")


def test_steepness_null_spectrum(freq):
    """A null spectrum has no steepness."""
    assert steepness(np.zeros(freq.size), freq.values) == 0.0


def test_steepness_fully_developed_sea(freq):
    """Steepness of a fully developed sea is 0.035 for any wind speed.

    The Pierson-Moskowitz sea is self-similar so its steepness is independent
    of the wind speed, which is what allows a single steepness threshold to
    define wind sea across the range of wind speeds.
    """
    values = []
    for wspd in [5, 10, 15, 20, 25]:
        efth = pierson_moskowitz(freq=freq, fp=0.13 * 9.81 / wspd, hs=0.0246 * wspd**2)
        values.append(steepness(efth.values, freq.values))
    assert np.allclose(values, 0.035, atol=1e-3)


def test_steepness_decreases_with_wave_age(freq):
    """Younger seas are steeper than older seas.

    Steepness is evaluated along the JONSWAP fetch-limited growth curve for a
    10 m/s wind, which is what makes it a proxy for the wave age criterion used
    by the other partitioning methods.
    """
    g = 9.81
    wspd = 10.0
    values = []
    for fetch in [1e4, 3e4, 1e5, 3e5]:
        fetch_scaled = g * fetch / wspd**2
        fp = max(3.5 * fetch_scaled**-0.33, 0.13) * g / wspd
        m0 = min(1.6e-7 * fetch_scaled, 3.64e-3) * wspd**4 / g**2
        efth = jonswap(freq=freq, fp=fp, hs=4 * np.sqrt(m0))
        values.append(steepness(efth.values, freq.values))
    assert np.all(np.diff(values) < 0)
    # The fully developed end of the growth curve is still above the default
    # min_steepness threshold used to classify wind sea
    assert values[-1] > 0.025


def test_steepness_swell_is_gentle(freq):
    """A typical swell falls well below the wind sea threshold."""
    efth = jonswap(freq=freq, fp=1 / 14.0, hs=1.5)
    assert steepness(efth.values, freq.values) < 0.01


def test_steepness_increases_in_shallow_water(freq):
    """Shoaling makes waves steeper as the local wavelength shortens."""
    efth = jonswap(freq=freq, fp=1 / 9.0, hs=2.0)
    deep = steepness(efth.values, freq.values)
    values = [steepness(efth.values, freq.values, dpt=d) for d in [1000, 50, 20, 10]]
    assert values[0] == pytest.approx(deep, rel=1e-3)
    assert np.all(np.diff(values) > 0)


def test_steepness_tail(freq):
    """The tail correction compensates for a truncated frequency grid."""
    efth = jonswap(freq=freq, fp=0.26, hs=0.6)
    full = steepness(efth.values, freq.values)
    # Truncate the grid at 0.4 Hz, cutting through the wind sea peak
    itrunc = np.searchsorted(freq.values, 0.4)
    trunc = efth.values[:itrunc]
    ftrunc = freq.values[:itrunc]
    assert steepness(trunc, ftrunc) < 0.9 * full
    assert steepness(trunc, ftrunc, tail=True) == pytest.approx(full, rel=0.1)


def test_steepness_tail_ignored_on_short_grids(freq):
    """The tail is not extrapolated when the grid does not resolve 0.333 Hz."""
    itrunc = np.searchsorted(freq.values, 0.3)
    trunc = jonswap(freq=freq, fp=1 / 12.0, hs=2.0).values[:itrunc]
    ftrunc = freq.values[:itrunc]
    assert steepness(trunc, ftrunc, tail=True) == steepness(trunc, ftrunc, tail=False)
