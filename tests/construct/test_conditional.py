"""Testing Gaussian fitting."""
import os
import numpy as np
import pytest

from wavespectra import read_swan
from wavespectra.construct.frequency import conditional, jonswap, gaussian


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def freq():
    filename = os.path.join(FILES_DIR, "swanfile.spec")
    _dset = read_swan(filename).interp({"freq": np.arange(0.04, 0.4, 0.001)})
    yield _dset.freq


def test_conditional(freq):
    """Test Hs, Tp values are conserved."""
    cond = True
    kwargs = dict(freq=freq, hs=2, fp=1 / 10, gamma=3.3, gw=0.07, cond=cond)
    ds_true = jonswap(**kwargs)
    ds_false = gaussian(**kwargs)
    ds = conditional(**kwargs)
    assert ds.equals(ds_true)
    kwargs['cond']=False
    ds = conditional(**kwargs)
    assert ds.equals(ds_false)
