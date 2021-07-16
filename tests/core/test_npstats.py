import os
import pytest
import numpy as np

from wavespectra import read_swan
from wavespectra.core.npstats import hs, dpm_gufunc, dp_gufunc, tps_gufunc, tp_gufunc, dpspr_gufunc


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


def test_dpm_gufunc(dset):
    ds = dset.isel(time=0, site=0, drop=True)
    ipeak = ds.efth.spec._peak(ds.spec.oned())
    momsin, momcos = ds.spec.momd(1)
    out = np.array([float(0)])
    out = dpm_gufunc(int(ipeak), momsin.values, momcos.values, out)
    assert np.isclose(out, 249.09263611)
    out = dpm_gufunc(0, momsin.values, momcos.values, out)
    assert np.isnan(out)


def test_dp_gufunc(dset):
    ds = dset.isel(time=0, site=0, drop=True)
    ipeak = np.int64(ds.efth.spec._peak(ds.spec.oned()))
    dir = ds.dir.values.astype("float32")
    out = np.array([float(0)]).astype("float32")
    out = dp_gufunc(ipeak, dir, out)
    assert np.isclose(out, 55)


def test_tps_gufunc(dset):
    ds = dset.isel(time=0, site=0, drop=True)
    ipeak = np.int64(ds.efth.spec._peak(ds.spec.oned()))
    spectrum = ds.spec.oned().values.astype("float64")
    freq = ds.freq.values.astype("float32")
    out = np.array([float(0)]).astype("float32")
    out = tps_gufunc(ipeak, spectrum, freq, out)
    assert np.isclose(out, 12.907742)
    out = tps_gufunc(0, spectrum, freq, out)
    assert np.isnan(out)


def test_tp_gufunc(dset):
    ds = dset.isel(time=0, site=0, drop=True)
    ipeak = np.int64(ds.efth.spec._peak(ds.spec.oned()))
    spectrum = ds.spec.oned().values.astype("float64")
    freq = ds.freq.values.astype("float32")
    out = np.array([float(0)]).astype("float32")
    out = tp_gufunc(ipeak, spectrum, freq, out)
    assert np.isclose(out, 13.568521)
    out = tp_gufunc(0, spectrum, freq, out)
    assert np.isnan(out)


def test_dpspr_gufunc(dset):
    ds = dset.isel(time=0, site=0, drop=True)
    ipeak = np.int64(ds.efth.spec._peak(ds.spec.oned()))
    fdspr1 = ds.spec.fdspr(mom=1).values.astype("float64")
    fdspr2 = ds.spec.fdspr(mom=2).values.astype("float64")

    out = np.array([float(0)]).astype("float32")
    out = dpspr_gufunc(ipeak, fdspr1, out)
    assert np.isclose(out, 8.463889)

    out = dpspr_gufunc(ipeak, fdspr2, out)
    assert np.isclose(out, 29.384691)

    out = dpspr_gufunc(0, fdspr1, out)
    assert np.isnan(out)
