import os
import pytest
import numpy as np

from wavespectra import read_swan
from wavespectra.core.xrstats import peak_wave_direction, mean_direction_at_peak_wave_period, peak_wave_period


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def dset():
    """Load SpecDset but skip test if matplotlib is not installed."""
    pytest.importorskip("matplotlib")
    dset = read_swan(os.path.join(FILES_DIR, "swanfile.spec"), as_site=True)
    return dset


def test_peak_wave_direction(dset):
    dp = peak_wave_direction(dset)
    dp = peak_wave_direction(dset.efth)
    with pytest.raises(ValueError):
        dp = peak_wave_direction(dset.spec.oned())


def test_mean_direction_at_peak_wave_period(dset):
    dpm = mean_direction_at_peak_wave_period(dset)
    dpm = mean_direction_at_peak_wave_period(dset.efth)
    with pytest.raises(ValueError):
        dpm = mean_direction_at_peak_wave_period(dset.spec.oned())


def test_peak_wave_period(dset):
    tp = peak_wave_period(dset)
    tp = peak_wave_period(dset.efth)
