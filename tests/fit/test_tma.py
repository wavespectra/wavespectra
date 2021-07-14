"""Testing Pierson-Moskowitz fitting."""
import os
import numpy as np
import pytest

from wavespectra import read_swan
from wavespectra.fit.tma import fit_tma
from wavespectra.fit.jonswap import fit_jonswap


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def freq():
    filename = os.path.join(FILES_DIR, "swanfile.spec")
    _dset = read_swan(filename).interp({"freq": np.arange(0.04, 0.4, 0.001)})
    yield _dset.freq


def test_hs_tp(freq):
    """Test Hs, Tp values are conserved."""
    ds1 = fit_tma(freq=freq, hs=2, tp=10, dep=10)
    ds2 = fit_tma(freq=freq, hs=2, tp=10, dep=50)
    assert pytest.approx(float(ds1.spec.hs()), 2)
    assert pytest.approx(float(ds1.spec.tp()), 10)
    assert pytest.approx(float(ds2.spec.hs()), 2)
    assert pytest.approx(float(ds2.spec.tp()), 10)


def test_jonswap_tma_deepwater_equal(freq):
    """Test TMA becomes Jonswap in deep water."""
    ds1 = fit_jonswap(freq=freq, hs=2, tp=10)
    ds2 = fit_tma(freq=freq, hs=2, tp=10, dep=80)
    assert np.allclose(ds1.values, ds2.values, rtol=1e6)


def test_freq_input_type(freq):
    """Test frequency input can also list, numpy or DataArray."""
    ds1 = fit_tma(freq=freq, hs=2, tp=10, dep=10.0)
    ds2 = fit_tma(freq=freq.values, hs=2, tp=10, dep=10.0)
    ds3 = fit_tma(freq=list(freq.values), hs=2, tp=10, dep=10.0)
    assert ds1.identical(ds2)
    assert ds1.identical(ds3)
