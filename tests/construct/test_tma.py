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
    ds1 = fit_tma(freq=freq, fp=0.1, dep=10, hs=2)
    ds2 = fit_tma(freq=freq, fp=0.1, dep=50, hs=2)
    assert float(ds1.spec.hs()) == pytest.approx(2, rel=1e3)
    assert float(ds1.spec.fp()) == pytest.approx(0.1, rel=1e3)
    assert float(ds2.spec.hs()) == pytest.approx(2, rel=1e3)
    assert float(ds2.spec.tp()) == pytest.approx(10, rel=1e3)


def test_jonswap_tma_deepwater_equal(freq):
    """Test TMA becomes Jonswap in deep water."""
    ds1 = fit_jonswap(freq=freq, fp=0.1, hs=2)
    ds2 = fit_tma(freq=freq, fp=0.1, dep=80, hs=2)
    assert np.allclose(ds1.values, ds2.values, rtol=1e6)


def test_freq_input_type(freq):
    """Test frequency input can also list, numpy or DataArray."""
    ds1 = fit_tma(freq=freq, fp=0.1, dep=10.0, hs=2)
    ds2 = fit_tma(freq=freq.values, fp=0.1, dep=10.0, hs=2)
    ds3 = fit_tma(freq=list(freq.values), fp=0.1, dep=10.0, hs=2)
    assert ds1.identical(ds2)
    assert ds1.identical(ds3)


def test_scaling(freq):
    """Test peak is higher for higher gamma."""
    ds1 = fit_tma(freq=freq, fp=0.1, dep=10.0, gamma=1.0, alpha=0.0081, hs=2)
    ds2 = fit_tma(freq=freq, fp=0.1, dep=10.0, gamma=3.3, alpha=0.0081)
    assert np.isclose(ds1.spec.hs(), 2)
    assert not np.isclose(ds2.spec.hs(), 2)
