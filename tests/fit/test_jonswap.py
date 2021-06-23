"""Testing Jonswap fitting."""
import os
import numpy as np
import pytest

from wavespectra import read_swan
from wavespectra.fit.jonswap import jonswap


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def dset():
    filename = os.path.join(FILES_DIR, "swanfile.spec")
    _dset = read_swan(filename)
    yield _dset


@pytest.fixture(scope="module")
def freq():
    filename = os.path.join(FILES_DIR, "swanfile.spec")
    _dset = read_swan(filename).interp({"freq": np.arange(0.04, 0.4, 0.001)})
    yield _dset.freq


def test_hs_tp(freq):
    """Test Hs, Tp values are conserved."""
    ds = jonswap(hs=2, tp=10, freq=freq, gamma=1.5)
    assert pytest.approx(float(ds.spec.hs()), 2)
    assert pytest.approx(float(ds.spec.tp()), 10)


def test_gamma(freq):
    """Test peak is higher for higher gamma."""
    ds1 = jonswap(hs=2, tp=10, freq=freq, gamma=1.0)
    ds2 = jonswap(hs=2, tp=10, freq=freq, gamma=3.3)
    assert ds2.max() > ds1.max()

