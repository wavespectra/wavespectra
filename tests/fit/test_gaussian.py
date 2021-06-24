"""Testing Gaussian fitting."""
import os
import numpy as np
import pytest

from wavespectra import read_swan
from wavespectra.fit.gaussian import gaussian


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def freq():
    filename = os.path.join(FILES_DIR, "swanfile.spec")
    _dset = read_swan(filename).interp({"freq": np.arange(0.04, 0.4, 0.001)})
    yield _dset.freq


def test_hs_tp(freq):
    """Test Hs, Tp values are conserved."""
    ds = gaussian(freq=freq, hs=2, fp=1/10, tm01=8, tm02=7.7)
    assert pytest.approx(float(ds.spec.hs()), 2)
    assert pytest.approx(float(ds.spec.tp()), 10)


def test_gamma(freq):
    """Test peak is higher when Tm and Tz are closer."""
    ds1 = gaussian(freq=freq, hs=2, fp=1/10, tm01=8.0, tm02=8.0)
    ds2 = gaussian(freq=freq, hs=2, fp=1/10, tm01=8.0, tm02=7.5)
    assert ds1.max() > ds2.max()

