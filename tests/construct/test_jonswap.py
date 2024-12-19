"""Testing Jonswap fitting."""

import os
import numpy as np
import pytest

from wavespectra import read_swan
from wavespectra.construct.frequency import jonswap


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def freq():
    filename = os.path.join(FILES_DIR, "swanfile.spec")
    _dset = read_swan(filename).interp({"freq": np.arange(0.04, 0.4, 0.001)})
    yield _dset.freq


def test_hs_fp(freq):
    """Test Hs, fp values are conserved."""
    ds = jonswap(freq=freq, fp=0.1, gamma=1.5, hs=2.0)
    assert float(ds.spec.hs()) == pytest.approx(2, rel=0.001)
    assert float(ds.spec.tp()) == pytest.approx(10, rel=0.001)


def test_gamma(freq):
    """Test peak is higher for higher gamma."""
    ds1 = jonswap(freq=freq, fp=0.1, gamma=1.0, hs=2)
    ds2 = jonswap(freq=freq, fp=0.1, gamma=3.3, hs=2)
    assert ds2.max() > ds1.max()


def test_freq_input_type(freq):
    """Test frequency input can also list, numpy or DataArray."""
    ds1 = jonswap(freq=freq, fp=0.1, gamma=1.0, hs=2)
    ds2 = jonswap(freq=freq.values, fp=0.1, gamma=1.0, hs=2)
    ds3 = jonswap(freq=list(freq.values), fp=0.1, gamma=1.0, hs=2)
    assert ds1.identical(ds2)
    assert ds1.identical(ds3)


def test_scaling(freq):
    """Test peak is higher for higher gamma."""
    ds1 = jonswap(freq=freq, fp=0.1, gamma=1.0, alpha=0.0081, hs=2)
    ds2 = jonswap(freq=freq, fp=0.1, gamma=3.3, alpha=0.0081)
    assert np.isclose(ds1.spec.hs(), 2)
    assert not np.isclose(ds2.spec.hs(), 2)
