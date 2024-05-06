"""Testing Pierson-Moskowitz fitting."""
import os
import numpy as np
import pytest

from wavespectra import read_swan
from wavespectra.construct.frequency import jonswap, pierson_moskowitz


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def freq():
    filename = os.path.join(FILES_DIR, "swanfile.spec")
    _dset = read_swan(filename).interp({"freq": np.arange(0.04, 0.4, 0.001)})
    yield _dset.freq


def test_hs_fp(freq):
    """Test Hs, Tp values are conserved."""
    ds = pierson_moskowitz(freq=freq, fp=0.1, hs=2)
    assert float(ds.spec.hs()) == pytest.approx(2, rel=1e3)
    assert float(ds.spec.tp()) == pytest.approx(10, rel=1e3)


def test_jonswap_gamma_1_equal(freq):
    """Test Jonswap becomes Pierson-Moskowitz when gamma <= 1."""
    ds1 = jonswap(freq=freq, fp=0.1, gamma=1.0, hs=2)
    ds2 = pierson_moskowitz(freq=freq, fp=0.1, hs=2)
    assert np.allclose(ds1.values, ds2.values, rtol=1e6)


def test_freq_input_type(freq):
    """Test frequency input can also list, numpy or DataArray."""
    ds1 = pierson_moskowitz(freq=freq, fp=0.1, hs=2)
    ds2 = pierson_moskowitz(freq=freq.values, fp=0.1, hs=2)
    ds3 = pierson_moskowitz(freq=list(freq.values), fp=0.1, hs=2)
    assert ds1.identical(ds2)
    assert ds1.identical(ds3)
