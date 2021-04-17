from wavespectra.construct import ochihubble
import numpy as np
from numpy.testing import assert_allclose
import os
import xarray as xr
import pytest

# ------------ define datasets ---------

from wavespectra import read_triaxys

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")

freqs = np.sort(1.5 / 1.2 ** np.arange(0, 25))
give_test_spectrum = ochihubble(
    hs=[1, 1.2],
    tp=[3, 20],
    dp=[180, 180],
    L=[1, 1],
    freqs=freqs,
    dspr=[0, 0],
)

triaxis_single = read_triaxys(os.path.join(FILES_DIR, "triaxys.DIRSPEC"))

one_day_later = triaxis_single.copy()
one_day_later["time"] = one_day_later.time + np.timedelta64(1, "D")
triaxis_two_days = xr.concat([triaxis_single, one_day_later], dim="time")

# ------------- test functions ----------------


def test_oned():
    test = give_test_spectrum
    test = test.spec.oned().spec.from_bins_to_continuous()
    assert_allclose(test.spec.hs(), np.sqrt(1 + 1.2 ** 2), atol=0.01)


def test_twod():
    test = give_test_spectrum
    test = test.spec.from_bins_to_continuous()
    assert_allclose(test.spec.hs(), np.sqrt(1 + 1.2 ** 2), atol=0.01)


def test_compare_1d_and_twod_squashed():
    test = give_test_spectrum
    test1 = test.spec.from_bins_to_continuous().spec.oned()
    test2 = test.spec.oned().spec.from_bins_to_continuous()

    assert_allclose(test1.data, test2.data, atol=1e-2)


def test_triaxis_single():
    _ = triaxis_single.spec.from_bins_to_continuous()


def test_triaxis_two_days():
    test1 = triaxis_two_days.spec.from_bins_to_continuous().spec.oned()
    test2 = triaxis_two_days.spec.oned().spec.from_bins_to_continuous()

    assert_allclose(test1.data, test2.data, atol=0.02)


def test_no_convergence():

    with pytest.raises(ValueError):
        give_test_spectrum.spec.from_bins_to_continuous(tolerance=1e-20)
